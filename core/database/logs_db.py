from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, BigInteger, Column, DateTime, Index, Integer, Numeric, String, select
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 (type hint)

from core.config import get_settings
from core.database.postgres_database import Base, PostgresDatabase


class UsageLogModel(Base):
    __tablename__ = "usage_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    app_id = Column(String, nullable=False, index=True)
    operation_type = Column(String, nullable=False)
    status = Column(String, nullable=False)
    duration_ms = Column(Numeric(10, 2))
    tokens_used = Column(Integer)
    log_metadata = Column("metadata", JSON)
    error = Column(String)

    __table_args__ = (Index("idx_usage_logs_app_user_ts", "app_id", "user_id", "timestamp"),)


class LogsDB:
    """Singleton wrapper around PostgresDatabase to store usage logs via ORM."""

    _instance: Optional["LogsDB"] = None

    @classmethod
    async def get(cls) -> "LogsDB":
        if cls._instance is None:
            cls._instance = LogsDB()
            await cls._instance._init()
        return cls._instance

    async def _init(self):
        settings = get_settings()
        self.db = PostgresDatabase(uri=settings.POSTGRES_URI)
        await self.db.initialize()
        async with self.db.engine.begin() as conn:
            await conn.run_sync(lambda c: UsageLogModel.__table__.create(c, checkfirst=True))

    # ------------------------------------------------------------------
    async def insert(self, row: Dict[str, Any]):
        # SQLAlchemy reserved 'metadata' attribute name, map to log_metadata
        row_db = row.copy()
        row_db["log_metadata"] = row_db.pop("metadata")
        async with self.db.async_session() as session:  # type: AsyncSession
            obj = UsageLogModel(**row_db)
            session.add(obj)
            await session.commit()

    async def query(
        self,
        user_id: str,
        app_id: str,
        limit: int = 100,
        since: Optional[str] = None,
        op_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        async with self.db.async_session() as session:  # type: AsyncSession
            stmt = (
                select(UsageLogModel)
                .where(UsageLogModel.user_id == user_id, UsageLogModel.app_id == app_id)
                .order_by(UsageLogModel.timestamp.desc())
                .limit(limit)
            )
            if since:
                stmt = stmt.where(UsageLogModel.timestamp >= datetime.fromisoformat(since))
            if op_type:
                stmt = stmt.where(UsageLogModel.operation_type == op_type)
            if status:
                stmt = stmt.where(UsageLogModel.status == status)

            rows = (await session.execute(stmt)).scalars().all()
            result: List[Dict[str, Any]] = []
            for r in rows:
                d = r.__dict__.copy()
                d["metadata"] = d.pop("log_metadata", None)
                result.append(d)
            return result

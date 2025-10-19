from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any, Dict, Optional

import httpx

"""Lightweight async wrapper around the Neon REST API used for programmatic project
and branch provisioning.  We intentionally keep the implementation minimal and only
implement the endpoints that are required for Morphik's *app-per-project* provisioning process.

The implementation purposefully avoids pulling in a heavy OpenAPI-generated client
and instead uses `httpx` which we already depend on.
"""

__all__ = ["NeonClient", "NeonAPIError"]


class NeonAPIError(RuntimeError):
    """Raised when the underlying Neon request returned an error status code."""

    def __init__(self, message: str, status_code: int | None = None, details: Any | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class NeonClient:
    """Async helper for the parts of the Neon V2 API that we need."""

    BASE_URL = "https://console.neon.tech/api/v2"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None

    # ---------------------------------------------------------------------
    # Low-level helpers
    # ---------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(  # pylint: disable=attribute-defined-outside-init
                base_url=self.BASE_URL,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=20.0,
            )
        return self._client

    async def _request(self, method: str, url: str, **kwargs) -> Any:  # noqa: ANN401
        client = await self._get_client()
        resp = await client.request(method, url, **kwargs)
        if resp.status_code >= 400:
            try:
                data = resp.json()
            except Exception:  # noqa: BLE001
                data = resp.text
            raise NeonAPIError("Neon API request failed", status_code=resp.status_code, details=data)
        if resp.status_code == 204:
            return None
        return resp.json()

    async def close(self) -> None:  # pragma: no cover – called during shutdown
        if self._client is not None:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # High-level operations
    # ------------------------------------------------------------------

    async def create_project(
        self, project_name: str, region: str = "aws-us-east-1", pg_version: int = 15
    ) -> Dict[str, Any]:
        """Create a *new* Neon project and return the parsed JSON response."""
        payload = {"project": {"name": project_name, "pg_version": pg_version, "region_id": region}}
        return await self._request("POST", "/projects", json=payload)

    async def create_branch(self, project_id: str, branch_name: str = "main") -> Dict[str, Any]:
        """Create a branch (and a read-write endpoint) for *project_id*."""
        payload = {
            "branch": {"name": branch_name},
            "endpoints": [{"type": "read_write"}],
        }
        return await self._request("POST", f"/projects/{project_id}/branches", json=payload)

    async def get_connection_uri(
        self,
        project_id: str,
        database_name: str,
        role_name: str,
        branch_id: Optional[str] = None,
    ) -> str:
        """Retrieve a fully-resolved PostgreSQL connection URI for the given database/role.

        If *branch_id* is omitted the project's *default* branch is used.
        """
        params = {"database_name": database_name, "role_name": role_name}
        if branch_id:
            params["branch_id"] = branch_id
        data = await self._request("GET", f"/projects/{project_id}/connection_uri", params=params)
        return data["connection_uri"]

    # ------------------------------------------------------------------
    # Convenience polling helpers
    # ------------------------------------------------------------------

    async def get_project(self, project_id: str) -> Dict[str, Any]:
        """Return the current project JSON (wrapper around GET)."""
        return await self._request("GET", f"/projects/{project_id}")

    async def wait_until_project_ready(
        self,
        project_id: str,
        *,
        timeout: float = 20.0,
        poll_interval: float = 0.5,
    ) -> None:
        """Block until the given project is *unlocked* or until *timeout*.

        Raises:
            NeonAPIError: If the project stays locked beyond *timeout*.
        """
        start = perf_counter()
        while True:
            data = await self.get_project(project_id)
            locked = data.get("project", {}).get("locked", False)
            if not locked:
                return

            if perf_counter() - start > timeout:
                raise NeonAPIError(
                    f"Project {project_id} did not unlock within {timeout} seconds",
                    details=data,
                )

            await asyncio.sleep(poll_interval)

    # Convenience helper -------------------------------------------------

    async def build_connection_string_from_branch(self, branch_resp: Dict[str, Any]) -> str:
        """Extract the connection URI from a *create branch* response.

        The Neon API returns a list of `connection_uris`.  We simply take the first one.
        """
        uris = branch_resp.get("connection_uris") or []
        if not uris:
            raise NeonAPIError("create_branch response did not include connection_uris", details=branch_resp)
        return uris[0]

    async def delete_project(self, project_id: str) -> None:
        """Delete an existing Neon *project_id* permanently.

        Neon responds with 202/204 on success.  We intentionally ignore the
        response body and rely on the status code check inside *_request*.
        """
        await self._request("DELETE", f"/projects/{project_id}")

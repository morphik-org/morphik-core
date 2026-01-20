from enum import Enum
from typing import Any, Dict, Optional


class AccountTier(str, Enum):
    """Available account tiers."""

    FREE = "free"
    PRO = "pro"
    TEAMS = "teams"
    SELF_HOSTED = "self_hosted"


# Tier limits definition - organized by API endpoint usage
TIER_LIMITS = {
    AccountTier.FREE: {
        # Application limits
        "app_limit": 1,  # Maximum number of applications
        # Storage limits
        "storage_file_limit": 30,  # Maximum number of files in storage
        "storage_size_limit_gb": 0.25,  # Maximum storage size in GB
        "ingest_limit": 200,  # Total pages that can be ingested (lifetime)
        # Query limits
        "hourly_query_limit": 30,  # Maximum queries per hour
        "monthly_query_limit": 50,  # Maximum queries per month
    },
    AccountTier.PRO: {
        # Application limits
        "app_limit": 5,  # Maximum number of applications
        # Storage limits
        "storage_file_limit": 1000,  # Maximum number of files in storage
        "storage_size_limit_gb": 2,  # Maximum storage size in GB
        "ingest_limit": 1500,
        # Query limits
        "hourly_query_limit": 100,  # Maximum queries per hour
        "monthly_query_limit": 10000,  # Maximum queries per month
    },
    AccountTier.TEAMS: {
        # Teams tier – generous limits but still bounded
        # Application limits
        "app_limit": 100,  # Maximum number of applications
        # Storage limits
        "storage_file_limit": 500000,  # Maximum number of files in storage
        "storage_size_limit_gb": 10,  # Maximum storage size in GB
        "ingest_limit": 1000000,
        # Query limits
        "hourly_query_limit": 500,  # Maximum queries per hour
        "monthly_query_limit": 50000,  # Maximum queries per month
    },
    AccountTier.SELF_HOSTED: {
        # Self-hosted has no limits
        # Application limits
        "app_limit": float("inf"),  # Maximum number of applications
        # Storage limits
        "storage_file_limit": float("inf"),  # Maximum number of files in storage
        "storage_size_limit_gb": float("inf"),  # Maximum storage size in GB
        "ingest_limit": float("inf"),
        # Query limits
        "hourly_query_limit": float("inf"),  # Maximum queries per hour
        "monthly_query_limit": float("inf"),  # Maximum queries per month
    },
}


def get_tier_limits(tier: AccountTier, custom_limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get limits for a specific account tier.

    Args:
        tier: The account tier
        custom_limits: Optional custom limits for CUSTOM tier

    Returns:
        Dict of limits for the specified tier
    """
    if tier == AccountTier.TEAMS and custom_limits:
        # Merge default custom limits with the provided custom limits
        return {**TIER_LIMITS[tier], **custom_limits}

    return TIER_LIMITS[tier]

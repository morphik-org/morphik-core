class DataBridgeError(Exception):
    """Base exception for DataBridge SDK"""

    pass


class AuthenticationError(DataBridgeError):
    """Authentication related errors"""

    pass


class ConnectionError(DataBridgeError):
    """Connection related errors"""

    pass

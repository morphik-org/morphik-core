class MorphikError(Exception):
    """Base exception for Morphik SDK"""

    pass


class AuthenticationError(MorphikError):
    """Authentication related errors"""

    pass


class ConnectionError(MorphikError):
    """Connection related errors"""

    pass

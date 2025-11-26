"""
Abstract Base Class for Authentication Providers

All auth providers (Okta, Azure, PyAuth) implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from ..models import AuthResult, AuthSession, UserInfo, TokenInfo, ApiKeys


class AuthProviderBase(ABC):
    """
    Abstract base class for authentication providers.

    Each provider implements OAuth2/OIDC flows specific to their platform.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'okta', 'azure', 'pyauth')"""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable provider name (e.g., 'Sign in with Okta')"""
        pass

    @abstractmethod
    async def get_authorization_url(self, state: str, redirect_uri: str) -> str:
        """
        Get the URL to redirect user for authentication.

        Args:
            state: Random state parameter for CSRF protection
            redirect_uri: URL to redirect back after authentication

        Returns:
            Authorization URL to redirect user to
        """
        pass

    @abstractmethod
    async def exchange_code_for_token(
        self,
        code: str,
        redirect_uri: str,
        state: Optional[str] = None
    ) -> AuthResult:
        """
        Exchange authorization code for access token.

        Called after user is redirected back from IdP.

        Args:
            code: Authorization code from callback
            redirect_uri: Same redirect_uri used in authorization request
            state: State parameter for verification

        Returns:
            AuthResult with session on success, or error details on failure
        """
        pass

    @abstractmethod
    async def refresh_token(self, session: AuthSession) -> AuthResult:
        """
        Refresh an expired access token.

        Args:
            session: Current session with refresh token

        Returns:
            AuthResult with updated session on success
        """
        pass

    @abstractmethod
    async def get_user_info(self, access_token: str) -> Optional[UserInfo]:
        """
        Get user information from the identity provider.

        Args:
            access_token: Valid access token

        Returns:
            UserInfo on success, None on failure
        """
        pass

    async def get_api_keys(self, session: AuthSession) -> Optional[ApiKeys]:
        """
        Get LLM API keys for the authenticated user.

        Override this method if your provider can supply API keys.
        Default implementation returns None (use environment variables).

        Args:
            session: Authenticated session

        Returns:
            ApiKeys on success, None if not available
        """
        return None

    async def validate_token(self, access_token: str) -> bool:
        """
        Validate that an access token is still valid.

        Default implementation tries to get user info.
        Override for more efficient validation.

        Args:
            access_token: Token to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            user_info = await self.get_user_info(access_token)
            return user_info is not None
        except Exception:
            return False

    async def logout(self, session: AuthSession) -> Optional[str]:
        """
        Perform provider-specific logout.

        Override to implement single sign-out.

        Args:
            session: Session to terminate

        Returns:
            Optional logout URL to redirect user to
        """
        return None

    def _create_session(
        self,
        user: UserInfo,
        token: TokenInfo,
        api_keys: Optional[ApiKeys] = None
    ) -> AuthSession:
        """
        Helper to create an AuthSession.

        Args:
            user: Authenticated user info
            token: OAuth tokens
            api_keys: Optional API keys

        Returns:
            New AuthSession instance
        """
        from datetime import datetime, timedelta

        # Default session expiry: 9 hours (configurable)
        from ..config import get_auth_config
        config = get_auth_config()
        expires_at = datetime.utcnow() + timedelta(hours=config.session.timeout_hours)

        return AuthSession(
            user=user,
            token=token,
            api_keys=api_keys,
            expires_at=expires_at,
            provider=self.name,
        )

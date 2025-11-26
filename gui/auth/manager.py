"""
Authentication Manager

Central orchestrator for authentication flows, session management,
and provider coordination.
"""

import logging
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from .config import AuthConfig, AuthProvider, get_auth_config
from .models import AuthResult, AuthSession, UserInfo, ApiKeys
from .session_store import SessionStore, create_session_store
from .providers.base import AuthProviderBase
from .providers.okta import OktaProvider
from .providers.azure import AzureProvider
from .providers.pyauth import PyAuthProvider, PyAuthProviderSimple

logger = logging.getLogger(__name__)


class AuthManager:
    """
    Central authentication manager.

    Coordinates between authentication providers and session storage.
    Handles the full authentication lifecycle including:
    - Provider selection and initialization
    - Login flow initiation
    - Callback handling
    - Session management
    - Token refresh
    - Logout
    """

    def __init__(self, config: Optional[AuthConfig] = None):
        """
        Initialize AuthManager.

        Args:
            config: Optional auth configuration. If not provided,
                    loads from environment via get_auth_config().
        """
        self.config = config or get_auth_config()
        self._providers: Dict[str, AuthProviderBase] = {}
        self._session_store: Optional[SessionStore] = None
        self._state_store: Dict[str, Dict[str, Any]] = {}  # CSRF state tracking
        self._initialized = False

    @property
    def is_enabled(self) -> bool:
        """Check if authentication is enabled"""
        return self.config.provider != AuthProvider.DISABLED

    @property
    def provider_name(self) -> str:
        """Get configured provider name"""
        return self.config.provider.value

    def initialize(self) -> None:
        """
        Initialize the auth manager.

        Creates session store and initializes configured providers.
        Call this at application startup.
        """
        if self._initialized:
            return

        if not self.is_enabled:
            logger.info("Authentication is disabled")
            self._initialized = True
            return

        # Initialize session store
        self._session_store = create_session_store(self.config.session)
        logger.info(f"Session store initialized: {self.config.session.store_type.value}")

        # Initialize configured provider
        self._init_provider()
        self._initialized = True
        logger.info(f"AuthManager initialized with provider: {self.provider_name}")

    def _init_provider(self) -> None:
        """Initialize the configured authentication provider"""
        provider = self.config.provider

        if provider == AuthProvider.OKTA and self.config.okta:
            self._providers["okta"] = OktaProvider(self.config.okta)
            logger.info("Okta provider initialized")

        elif provider == AuthProvider.AZURE and self.config.azure:
            self._providers["azure"] = AzureProvider(self.config.azure)
            logger.info("Azure AD provider initialized")

        elif provider == AuthProvider.PYAUTH and self.config.pyauth:
            # Use full PyAuthProvider or simple version based on config
            if self.config.pyauth.use_simple_provider:
                self._providers["pyauth"] = PyAuthProviderSimple(self.config.pyauth)
                logger.info("PyAuth simple provider initialized")
            else:
                self._providers["pyauth"] = PyAuthProvider(self.config.pyauth)
                logger.info("PyAuth provider initialized")

    def get_provider(self) -> Optional[AuthProviderBase]:
        """Get the active authentication provider"""
        if not self.is_enabled:
            return None
        return self._providers.get(self.provider_name)

    async def get_session_store(self) -> SessionStore:
        """Get the session store"""
        if not self._session_store:
            raise RuntimeError("AuthManager not initialized. Call initialize() first.")
        return self._session_store

    # -------------------------------------------------------------------------
    # Login Flow
    # -------------------------------------------------------------------------

    async def start_login(self, redirect_uri: str) -> AuthResult:
        """
        Start the login flow.

        Generates state for CSRF protection and returns authorization URL.

        Args:
            redirect_uri: Callback URL after authentication

        Returns:
            AuthResult with redirect_url to send user to IdP
        """
        provider = self.get_provider()
        if not provider:
            return AuthResult.failure(
                error="auth_disabled",
                description="Authentication is not enabled"
            )

        # Generate CSRF state
        state = secrets.token_urlsafe(32)

        # Store state for verification
        self._state_store[state] = {
            "redirect_uri": redirect_uri,
            "created_at": datetime.utcnow(),
            "provider": provider.name,
        }

        # Clean up old states (older than 10 minutes)
        self._cleanup_states()

        try:
            auth_url = await provider.get_authorization_url(state, redirect_uri)
            return AuthResult(
                success=True,
                redirect_url=auth_url,
            )
        except Exception as e:
            logger.exception("Failed to generate authorization URL")
            return AuthResult.failure(
                error="authorization_url_failed",
                description=str(e)
            )

    async def handle_callback(
        self,
        code: str,
        state: str,
        redirect_uri: str,
    ) -> AuthResult:
        """
        Handle OAuth callback after user authentication.

        Validates state, exchanges code for token, creates session.

        Args:
            code: Authorization code from IdP
            state: State parameter for CSRF verification
            redirect_uri: Same redirect_uri used in authorization request

        Returns:
            AuthResult with session on success
        """
        # Verify state
        state_data = self._state_store.pop(state, None)
        if not state_data:
            return AuthResult.failure(
                error="invalid_state",
                description="Invalid or expired state parameter"
            )

        # Check state hasn't expired (10 minute window)
        created_at = state_data.get("created_at", datetime.min)
        if datetime.utcnow() - created_at > timedelta(minutes=10):
            return AuthResult.failure(
                error="state_expired",
                description="Authentication request expired. Please try again."
            )

        # Get provider
        provider = self.get_provider()
        if not provider:
            return AuthResult.failure(
                error="provider_not_found",
                description="Authentication provider not configured"
            )

        # Exchange code for token
        result = await provider.exchange_code_for_token(code, redirect_uri, state)

        if result.success and result.session:
            # Store session
            store = await self.get_session_store()
            await store.set(result.session)

            # Try to get API keys
            if not result.session.api_keys:
                api_keys = await provider.get_api_keys(result.session)
                if api_keys:
                    result.session.api_keys = api_keys
                    await store.set(result.session)

            logger.info(f"User authenticated: {result.session.user.email}")

        return result

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    async def get_session(self, session_id: str) -> Optional[AuthSession]:
        """
        Retrieve a session by ID.

        Args:
            session_id: Session identifier (from cookie)

        Returns:
            AuthSession if valid, None otherwise
        """
        if not self.is_enabled:
            return None

        store = await self.get_session_store()
        session = await store.get(session_id)

        if session and session.is_valid:
            return session

        return None

    async def validate_session(self, session_id: str) -> Optional[AuthSession]:
        """
        Validate and potentially refresh a session.

        Checks if session is valid. If token is expired but refresh
        token is available, attempts to refresh.

        Args:
            session_id: Session identifier

        Returns:
            Valid AuthSession or None
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        # Check if token needs refresh
        if session.token and session.token.is_expired:
            logger.info(f"Token expired for session {session_id}, attempting refresh")
            result = await self.refresh_session(session)
            if result.success:
                return result.session
            else:
                # Refresh failed, invalidate session
                await self.logout(session_id)
                return None

        return session

    async def refresh_session(self, session: AuthSession) -> AuthResult:
        """
        Refresh an expired session token.

        Args:
            session: Session with expired token

        Returns:
            AuthResult with refreshed session
        """
        provider = self.get_provider()
        if not provider:
            return AuthResult.failure(
                error="provider_not_found",
                description="Authentication provider not configured"
            )

        result = await provider.refresh_token(session)

        if result.success and result.session:
            # Update stored session
            store = await self.get_session_store()
            await store.set(result.session)
            logger.info(f"Session refreshed: {session.session_id}")

        return result

    async def logout(self, session_id: str) -> Optional[str]:
        """
        Log out a session.

        Removes session from store and optionally returns
        IdP logout URL for single sign-out.

        Args:
            session_id: Session to terminate

        Returns:
            Optional logout URL to redirect user to
        """
        store = await self.get_session_store()
        session = await store.get(session_id)

        if session:
            # Delete from store
            await store.delete(session_id)
            logger.info(f"Session terminated: {session_id}")

            # Get provider logout URL
            provider = self.get_provider()
            if provider:
                return await provider.logout(session)

        return None

    # -------------------------------------------------------------------------
    # API Keys
    # -------------------------------------------------------------------------

    async def get_api_keys(self, session_id: str) -> Optional[ApiKeys]:
        """
        Get API keys for a session.

        First checks session cache, then fetches from provider if needed.

        Args:
            session_id: Session identifier

        Returns:
            ApiKeys if available
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        # Return cached keys if available
        if session.api_keys:
            return session.api_keys

        # Try to fetch from provider
        provider = self.get_provider()
        if provider:
            api_keys = await provider.get_api_keys(session)
            if api_keys:
                # Cache in session
                session.api_keys = api_keys
                store = await self.get_session_store()
                await store.set(session)
                return api_keys

        return None

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _cleanup_states(self) -> None:
        """Remove expired state entries"""
        cutoff = datetime.utcnow() - timedelta(minutes=10)
        expired = [
            state for state, data in self._state_store.items()
            if data.get("created_at", datetime.min) < cutoff
        ]
        for state in expired:
            del self._state_store[state]

    async def cleanup_sessions(self) -> int:
        """
        Clean up expired sessions.

        Call this periodically (e.g., via background task).

        Returns:
            Number of sessions removed
        """
        if not self._session_store:
            return 0
        return await self._session_store.cleanup_expired()


# Global instance (singleton pattern)
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """
    Get the global AuthManager instance.

    Creates and initializes if not exists.
    """
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
        _auth_manager.initialize()
    return _auth_manager


def reset_auth_manager() -> None:
    """Reset the global AuthManager (useful for testing)"""
    global _auth_manager
    _auth_manager = None

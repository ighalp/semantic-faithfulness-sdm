"""
Okta OIDC Authentication Provider

Implements OAuth2/OIDC flow for Okta identity provider.
"""

import logging
from typing import Optional
from urllib.parse import urlencode

from .base import AuthProviderBase
from ..models import AuthResult, AuthSession, UserInfo, TokenInfo, ApiKeys
from ..config import OktaConfig

logger = logging.getLogger(__name__)


class OktaProvider(AuthProviderBase):
    """
    Okta OIDC authentication provider.

    Implements standard Authorization Code flow with PKCE.
    """

    def __init__(self, config: OktaConfig):
        self.config = config
        self._http_client = None

    @property
    def name(self) -> str:
        return "okta"

    @property
    def display_name(self) -> str:
        return "Sign in with Okta"

    def _get_http_client(self):
        """Lazy initialization of HTTP client"""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.AsyncClient(timeout=30.0)
            except ImportError:
                raise ImportError(
                    "Okta provider requires 'httpx' package. "
                    "Install with: pip install httpx"
                )
        return self._http_client

    @property
    def _authorization_endpoint(self) -> str:
        return f"{self.config.issuer_url}/v1/authorize"

    @property
    def _token_endpoint(self) -> str:
        return f"{self.config.issuer_url}/v1/token"

    @property
    def _userinfo_endpoint(self) -> str:
        return f"{self.config.issuer_url}/v1/userinfo"

    @property
    def _logout_endpoint(self) -> str:
        return f"{self.config.issuer_url}/v1/logout"

    async def get_authorization_url(self, state: str, redirect_uri: str) -> str:
        """Build Okta authorization URL"""
        params = {
            "client_id": self.config.client_id,
            "response_type": "code",
            "scope": " ".join(self.config.scopes),
            "redirect_uri": redirect_uri,
            "state": state,
        }
        return f"{self._authorization_endpoint}?{urlencode(params)}"

    async def exchange_code_for_token(
        self,
        code: str,
        redirect_uri: str,
        state: Optional[str] = None
    ) -> AuthResult:
        """Exchange authorization code for tokens"""
        client = self._get_http_client()

        try:
            response = await client.post(
                self._token_endpoint,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": self.config.client_id,
                    "client_secret": self.config.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            if response.status_code != 200:
                error_data = response.json()
                return AuthResult.failure(
                    error=error_data.get("error", "token_exchange_failed"),
                    description=error_data.get("error_description", response.text)
                )

            token_data = response.json()
            token = TokenInfo.from_oauth_response(token_data)

            # Get user info
            user = await self.get_user_info(token.access_token)
            if not user:
                return AuthResult.failure(
                    error="userinfo_failed",
                    description="Failed to retrieve user information"
                )

            # Create session
            session = self._create_session(user, token)
            return AuthResult.authenticated(session)

        except Exception as e:
            logger.exception("Okta token exchange failed")
            return AuthResult.failure(
                error="token_exchange_error",
                description=str(e)
            )

    async def refresh_token(self, session: AuthSession) -> AuthResult:
        """Refresh expired access token"""
        if not session.token or not session.token.refresh_token:
            return AuthResult.failure(
                error="no_refresh_token",
                description="No refresh token available"
            )

        client = self._get_http_client()

        try:
            response = await client.post(
                self._token_endpoint,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": session.token.refresh_token,
                    "client_id": self.config.client_id,
                    "client_secret": self.config.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            if response.status_code != 200:
                error_data = response.json()
                return AuthResult.failure(
                    error=error_data.get("error", "refresh_failed"),
                    description=error_data.get("error_description", response.text)
                )

            token_data = response.json()
            new_token = TokenInfo.from_oauth_response(token_data)

            # Keep existing refresh token if not returned
            if not new_token.refresh_token:
                new_token.refresh_token = session.token.refresh_token

            # Update session with new token
            session.token = new_token
            return AuthResult.authenticated(session)

        except Exception as e:
            logger.exception("Okta token refresh failed")
            return AuthResult.failure(
                error="refresh_error",
                description=str(e)
            )

    async def get_user_info(self, access_token: str) -> Optional[UserInfo]:
        """Get user info from Okta userinfo endpoint"""
        client = self._get_http_client()

        try:
            response = await client.get(
                self._userinfo_endpoint,
                headers={"Authorization": f"Bearer {access_token}"}
            )

            if response.status_code != 200:
                logger.error(f"Okta userinfo failed: {response.text}")
                return None

            claims = response.json()
            return UserInfo.from_claims(claims)

        except Exception as e:
            logger.exception("Okta userinfo request failed")
            return None

    async def logout(self, session: AuthSession) -> Optional[str]:
        """Generate Okta logout URL"""
        if session.token and session.token.id_token:
            params = {
                "id_token_hint": session.token.id_token,
                "post_logout_redirect_uri": self.config.redirect_uri.replace("/callback", ""),
            }
            return f"{self._logout_endpoint}?{urlencode(params)}"
        return None

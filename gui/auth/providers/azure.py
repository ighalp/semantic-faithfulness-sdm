"""
Azure AD OIDC Authentication Provider

Implements OAuth2/OIDC flow for Microsoft Azure Active Directory.
"""

import logging
from typing import Optional
from urllib.parse import urlencode

from .base import AuthProviderBase
from ..models import AuthResult, AuthSession, UserInfo, TokenInfo, ApiKeys
from ..config import AzureConfig

logger = logging.getLogger(__name__)


class AzureProvider(AuthProviderBase):
    """
    Azure AD OIDC authentication provider.

    Implements standard Authorization Code flow for Microsoft identity platform.
    Supports both Azure AD (organizational) and Azure AD B2C.
    """

    def __init__(self, config: AzureConfig):
        self.config = config
        self._http_client = None

    @property
    def name(self) -> str:
        return "azure"

    @property
    def display_name(self) -> str:
        return "Sign in with Microsoft"

    def _get_http_client(self):
        """Lazy initialization of HTTP client"""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.AsyncClient(timeout=30.0)
            except ImportError:
                raise ImportError(
                    "Azure provider requires 'httpx' package. "
                    "Install with: pip install httpx"
                )
        return self._http_client

    @property
    def _base_url(self) -> str:
        return f"https://login.microsoftonline.com/{self.config.tenant_id}/oauth2/v2.0"

    @property
    def _authorization_endpoint(self) -> str:
        return f"{self._base_url}/authorize"

    @property
    def _token_endpoint(self) -> str:
        return f"{self._base_url}/token"

    @property
    def _logout_endpoint(self) -> str:
        return f"{self._base_url}/logout"

    @property
    def _graph_userinfo_endpoint(self) -> str:
        return "https://graph.microsoft.com/v1.0/me"

    async def get_authorization_url(self, state: str, redirect_uri: str) -> str:
        """Build Azure AD authorization URL"""
        params = {
            "client_id": self.config.client_id,
            "response_type": "code",
            "scope": " ".join(self.config.scopes),
            "redirect_uri": redirect_uri,
            "state": state,
            "response_mode": "query",
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
                    "scope": " ".join(self.config.scopes),
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

            # Get user info from Microsoft Graph
            user = await self.get_user_info(token.access_token)
            if not user:
                # Try to extract from id_token claims
                user = self._extract_user_from_id_token(token.id_token)

            if not user:
                return AuthResult.failure(
                    error="userinfo_failed",
                    description="Failed to retrieve user information"
                )

            # Create session
            session = self._create_session(user, token)
            return AuthResult.authenticated(session)

        except Exception as e:
            logger.exception("Azure token exchange failed")
            return AuthResult.failure(
                error="token_exchange_error",
                description=str(e)
            )

    def _extract_user_from_id_token(self, id_token: str) -> Optional[UserInfo]:
        """Extract user info from JWT id_token claims"""
        if not id_token:
            return None

        try:
            # Decode JWT without verification (we trust Azure's signature)
            import base64
            import json

            # Split token and get payload
            parts = id_token.split(".")
            if len(parts) != 3:
                return None

            # Decode payload (add padding if needed)
            payload = parts[1]
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += "=" * padding

            claims = json.loads(base64.urlsafe_b64decode(payload))
            return UserInfo.from_claims(claims)

        except Exception as e:
            logger.warning(f"Failed to decode id_token: {e}")
            return None

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
                    "scope": " ".join(self.config.scopes),
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

            # Azure usually returns new refresh token
            if not new_token.refresh_token:
                new_token.refresh_token = session.token.refresh_token

            # Update session with new token
            session.token = new_token
            return AuthResult.authenticated(session)

        except Exception as e:
            logger.exception("Azure token refresh failed")
            return AuthResult.failure(
                error="refresh_error",
                description=str(e)
            )

    async def get_user_info(self, access_token: str) -> Optional[UserInfo]:
        """Get user info from Microsoft Graph API"""
        client = self._get_http_client()

        try:
            response = await client.get(
                self._graph_userinfo_endpoint,
                headers={"Authorization": f"Bearer {access_token}"}
            )

            if response.status_code != 200:
                logger.warning(f"Microsoft Graph userinfo failed: {response.text}")
                return None

            data = response.json()

            # Map Microsoft Graph fields to our UserInfo
            claims = {
                "sub": data.get("id"),
                "email": data.get("mail") or data.get("userPrincipalName"),
                "name": data.get("displayName"),
                "given_name": data.get("givenName"),
                "family_name": data.get("surname"),
                "preferred_username": data.get("userPrincipalName"),
            }

            return UserInfo.from_claims(claims)

        except Exception as e:
            logger.exception("Microsoft Graph userinfo request failed")
            return None

    async def logout(self, session: AuthSession) -> Optional[str]:
        """Generate Azure AD logout URL"""
        params = {
            "post_logout_redirect_uri": self.config.redirect_uri.replace("/callback", ""),
        }
        return f"{self._logout_endpoint}?{urlencode(params)}"

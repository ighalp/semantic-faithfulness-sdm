"""
PyAuth OIDC Authentication Provider

Wrapper for internal PyAuthClient module (extends requests-oauthlib).
Provides OAuth2 authentication with corporate identity provider.
"""

import logging
from typing import Optional
from datetime import datetime, timedelta

from .base import AuthProviderBase
from ..models import AuthResult, AuthSession, UserInfo, TokenInfo, ApiKeys
from ..config import PyAuthConfig

logger = logging.getLogger(__name__)


class PyAuthProvider(AuthProviderBase):
    """
    PyAuth authentication provider.

    Wraps the internal PyAuthClient module which extends requests-oauthlib.
    Supports OAuth2 flows including Resource Owner Password Credentials (ROPC)
    and Authorization Code flow with Kerberos ticket integration.

    Configuration is loaded from environment variables (PYAUTH_*).
    """

    def __init__(self, config: PyAuthConfig):
        self.config = config
        self._client = None
        self._http_client = None

    @property
    def name(self) -> str:
        return "pyauth"

    @property
    def display_name(self) -> str:
        return "Sign in with Corporate SSO"

    def _get_pyauth_client(self):
        """
        Lazy initialization of PyAuthClient.

        TODO: YOUR INTEGRATION
        Import and configure your PyAuthClient here.
        The PyAuthClient extends requests-oauthlib.
        """
        if self._client is None:
            try:
                # TODO: YOUR INTEGRATION - Import your PyAuthClient module
                # from pyauthclient import PyAuthClient
                #
                # Example initialization based on provided config:
                # self._client = PyAuthClient(
                #     secure_endpoint=self.config.secure_endpoint,
                #     access_url=self.config.access_url,
                #     cons_app_id=self.config.cons_app_id,
                #     client_id=self.config.client_id,
                #     client_secret=self.config.client_secret,
                #     username=self.config.username,
                #     password=self.config.password,
                #     grant_type=self.config.grant_type,
                #     scope=self.config.scope,
                #     ssl_context=self._create_ssl_context(),
                #     proxies=self._get_proxies(),
                # )

                raise NotImplementedError(
                    "PyAuthClient integration not configured. "
                    "Please implement _get_pyauth_client() with your PyAuthClient module."
                )
            except ImportError:
                raise ImportError(
                    "PyAuth provider requires 'pyauthclient' module. "
                    "Contact your system administrator for installation."
                )
        return self._client

    def _get_http_client(self):
        """Lazy initialization of HTTP client for API calls"""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.AsyncClient(
                    timeout=30.0,
                    verify=self.config.ca_bundle_path or True,
                )
            except ImportError:
                raise ImportError(
                    "PyAuth provider requires 'httpx' package. "
                    "Install with: pip install httpx"
                )
        return self._http_client

    def _get_proxies(self) -> Optional[dict]:
        """Build proxy configuration if provided"""
        if self.config.proxy_host:
            proxy_url = f"http://{self.config.proxy_host}"
            if self.config.proxy_port:
                proxy_url = f"{proxy_url}:{self.config.proxy_port}"
            return {
                "http": proxy_url,
                "https": proxy_url,
            }
        return None

    def _create_ssl_context(self):
        """
        Create SSL context for PyAuthClient.

        TODO: YOUR INTEGRATION
        Configure SSL/TLS settings as required by your environment.
        """
        import ssl
        ctx = ssl.create_default_context()
        if self.config.ca_bundle_path:
            ctx.load_verify_locations(self.config.ca_bundle_path)
        return ctx

    async def get_authorization_url(self, state: str, redirect_uri: str) -> str:
        """
        Get authorization URL for browser redirect.

        For ROPC (password) grant, this returns None as no browser redirect is needed.
        For authorization code flow, returns the IdP authorization URL.

        TODO: YOUR INTEGRATION
        Implement based on your PyAuthClient's authorization flow.
        """
        if self.config.grant_type == "password":
            # ROPC doesn't use browser redirect
            # Return a local endpoint that will handle username/password form
            return f"/auth/pyauth/login?state={state}&redirect_uri={redirect_uri}"

        # Authorization code flow
        # TODO: YOUR INTEGRATION - Use PyAuthClient to build authorization URL
        # client = self._get_pyauth_client()
        # auth_url, _ = client.authorization_url(
        #     self.config.authorization_endpoint,
        #     state=state,
        #     redirect_uri=redirect_uri,
        # )
        # return auth_url

        raise NotImplementedError(
            "PyAuth authorization URL generation not configured. "
            "Please implement get_authorization_url() for your flow."
        )

    async def exchange_code_for_token(
        self,
        code: str,
        redirect_uri: str,
        state: Optional[str] = None
    ) -> AuthResult:
        """
        Exchange authorization code for tokens.

        For ROPC grant, 'code' parameter contains username:password.
        For authorization code flow, exchanges the code with the token endpoint.

        TODO: YOUR INTEGRATION
        Implement token exchange using your PyAuthClient.
        """
        try:
            if self.config.grant_type == "password":
                return await self._ropc_authenticate(code)

            # Authorization code flow
            # TODO: YOUR INTEGRATION
            # client = self._get_pyauth_client()
            # token = client.fetch_token(
            #     self.config.token_endpoint,
            #     code=code,
            #     redirect_uri=redirect_uri,
            # )
            # token_info = TokenInfo(
            #     access_token=token['access_token'],
            #     refresh_token=token.get('refresh_token'),
            #     token_type=token.get('token_type', 'Bearer'),
            #     expires_at=datetime.utcnow() + timedelta(seconds=token.get('expires_in', 3600)),
            #     id_token=token.get('id_token'),
            # )

            raise NotImplementedError(
                "PyAuth token exchange not configured. "
                "Please implement exchange_code_for_token() for your flow."
            )

        except Exception as e:
            logger.exception("PyAuth token exchange failed")
            return AuthResult.failure(
                error="token_exchange_error",
                description=str(e)
            )

    async def _ropc_authenticate(self, credentials: str) -> AuthResult:
        """
        Authenticate using Resource Owner Password Credentials.

        TODO: YOUR INTEGRATION
        Implement ROPC authentication using your PyAuthClient.

        Args:
            credentials: Format "username:password" or just use config values
        """
        try:
            # Parse credentials if provided, otherwise use config
            username = self.config.username
            password = self.config.password

            if credentials and ":" in credentials:
                parts = credentials.split(":", 1)
                username = parts[0]
                password = parts[1]

            if not username or not password:
                return AuthResult.failure(
                    error="missing_credentials",
                    description="Username and password are required"
                )

            # TODO: YOUR INTEGRATION - Use PyAuthClient for ROPC
            # client = self._get_pyauth_client()
            # token = client.fetch_token(
            #     self.config.token_endpoint,
            #     grant_type='password',
            #     username=username,
            #     password=password,
            #     scope=self.config.scope,
            # )
            #
            # token_info = TokenInfo(
            #     access_token=token['access_token'],
            #     refresh_token=token.get('refresh_token'),
            #     token_type=token.get('token_type', 'Bearer'),
            #     expires_at=datetime.utcnow() + timedelta(seconds=token.get('expires_in', 3600)),
            # )
            #
            # # Get user info
            # user = await self.get_user_info(token_info.access_token)
            # if not user:
            #     # Create minimal user from username
            #     user = UserInfo(id=username, email=username, name=username)
            #
            # # Get API keys if endpoint configured
            # api_keys = await self._fetch_api_keys(token_info.access_token)
            #
            # session = self._create_session(user, token_info, api_keys)
            # return AuthResult.authenticated(session)

            raise NotImplementedError(
                "PyAuth ROPC authentication not configured. "
                "Please implement _ropc_authenticate() with your PyAuthClient."
            )

        except Exception as e:
            logger.exception("PyAuth ROPC authentication failed")
            return AuthResult.failure(
                error="authentication_error",
                description=str(e)
            )

    async def refresh_token(self, session: AuthSession) -> AuthResult:
        """
        Refresh expired access token.

        TODO: YOUR INTEGRATION
        Implement token refresh using your PyAuthClient.
        """
        if not session.token or not session.token.refresh_token:
            return AuthResult.failure(
                error="no_refresh_token",
                description="No refresh token available"
            )

        try:
            # TODO: YOUR INTEGRATION
            # client = self._get_pyauth_client()
            # token = client.refresh_token(
            #     self.config.token_endpoint,
            #     refresh_token=session.token.refresh_token,
            # )
            #
            # new_token = TokenInfo(
            #     access_token=token['access_token'],
            #     refresh_token=token.get('refresh_token', session.token.refresh_token),
            #     token_type=token.get('token_type', 'Bearer'),
            #     expires_at=datetime.utcnow() + timedelta(seconds=token.get('expires_in', 3600)),
            # )
            #
            # session.token = new_token
            # return AuthResult.authenticated(session)

            raise NotImplementedError(
                "PyAuth token refresh not configured. "
                "Please implement refresh_token() with your PyAuthClient."
            )

        except Exception as e:
            logger.exception("PyAuth token refresh failed")
            return AuthResult.failure(
                error="refresh_error",
                description=str(e)
            )

    async def get_user_info(self, access_token: str) -> Optional[UserInfo]:
        """
        Get user info from the identity provider.

        TODO: YOUR INTEGRATION
        Implement userinfo retrieval based on your IdP's endpoint.
        """
        if not self.config.userinfo_endpoint:
            logger.debug("No userinfo endpoint configured for PyAuth")
            return None

        client = self._get_http_client()

        try:
            response = await client.get(
                self.config.userinfo_endpoint,
                headers={"Authorization": f"Bearer {access_token}"}
            )

            if response.status_code != 200:
                logger.warning(f"PyAuth userinfo failed: {response.text}")
                return None

            claims = response.json()
            return UserInfo.from_claims(claims)

        except Exception as e:
            logger.exception("PyAuth userinfo request failed")
            return None

    async def get_api_keys(self, session: AuthSession) -> Optional[ApiKeys]:
        """
        Get LLM API keys for the authenticated user.

        Calls the configured API keys endpoint to retrieve keys
        authorized for this user.
        """
        if not session.token:
            return None

        return await self._fetch_api_keys(session.token.access_token)

    async def _fetch_api_keys(self, access_token: str) -> Optional[ApiKeys]:
        """
        Fetch API keys from configured endpoint.

        TODO: YOUR INTEGRATION
        Adapt response parsing to match your API keys endpoint format.

        Expected response format:
        {
            "openai": "sk-...",
            "anthropic": "sk-ant-...",
            "gemini": "AIza...",
            ...
        }
        """
        if not self.config.api_keys_endpoint:
            return None

        client = self._get_http_client()

        try:
            response = await client.get(
                self.config.api_keys_endpoint,
                headers={"Authorization": f"Bearer {access_token}"}
            )

            if response.status_code != 200:
                logger.warning(f"API keys fetch failed: {response.status_code}")
                return None

            data = response.json()

            # TODO: YOUR INTEGRATION - Adapt to your API response format
            return ApiKeys(
                openai=data.get("openai"),
                anthropic=data.get("anthropic"),
                gemini=data.get("gemini"),
                extra={k: v for k, v in data.items()
                       if k not in ("openai", "anthropic", "gemini")}
            )

        except Exception as e:
            logger.exception("API keys fetch failed")
            return None

    async def logout(self, session: AuthSession) -> Optional[str]:
        """
        Perform logout / token revocation.

        TODO: YOUR INTEGRATION
        Implement logout based on your IdP's requirements.
        """
        if not self.config.logout_endpoint:
            return None

        # TODO: YOUR INTEGRATION - Revoke token if supported
        # client = self._get_pyauth_client()
        # if session.token:
        #     try:
        #         client.revoke_token(
        #             self.config.logout_endpoint,
        #             token=session.token.access_token,
        #         )
        #     except Exception as e:
        #         logger.warning(f"Token revocation failed: {e}")

        # Return logout redirect URL if configured
        if self.config.post_logout_redirect_uri:
            return self.config.post_logout_redirect_uri

        return None


class PyAuthProviderSimple(AuthProviderBase):
    """
    Simplified PyAuth provider using requests-oauthlib directly.

    Use this if you want to integrate without the full PyAuthClient module.
    Implements standard OAuth2 Authorization Code flow.
    """

    def __init__(self, config: PyAuthConfig):
        self.config = config
        self._oauth_session = None

    @property
    def name(self) -> str:
        return "pyauth_simple"

    @property
    def display_name(self) -> str:
        return "Sign in with Corporate SSO"

    def _get_oauth_session(self):
        """Create OAuth2Session from requests-oauthlib"""
        if self._oauth_session is None:
            try:
                from requests_oauthlib import OAuth2Session

                self._oauth_session = OAuth2Session(
                    client_id=self.config.client_id,
                    redirect_uri=self.config.redirect_uri,
                    scope=self.config.scope.split() if self.config.scope else None,
                )
            except ImportError:
                raise ImportError(
                    "PyAuth simple provider requires 'requests-oauthlib' package. "
                    "Install with: pip install requests-oauthlib"
                )
        return self._oauth_session

    async def get_authorization_url(self, state: str, redirect_uri: str) -> str:
        """Get authorization URL using requests-oauthlib"""
        if not self.config.authorization_endpoint:
            raise ValueError("authorization_endpoint not configured")

        oauth = self._get_oauth_session()
        oauth.redirect_uri = redirect_uri

        auth_url, _ = oauth.authorization_url(
            self.config.authorization_endpoint,
            state=state,
        )
        return auth_url

    async def exchange_code_for_token(
        self,
        code: str,
        redirect_uri: str,
        state: Optional[str] = None
    ) -> AuthResult:
        """Exchange code using requests-oauthlib"""
        if not self.config.token_endpoint:
            return AuthResult.failure(
                error="configuration_error",
                description="token_endpoint not configured"
            )

        try:
            oauth = self._get_oauth_session()
            oauth.redirect_uri = redirect_uri

            # Build callback URL with code
            callback_url = f"{redirect_uri}?code={code}"
            if state:
                callback_url += f"&state={state}"

            token = oauth.fetch_token(
                self.config.token_endpoint,
                authorization_response=callback_url,
                client_secret=self.config.client_secret,
            )

            token_info = TokenInfo(
                access_token=token['access_token'],
                refresh_token=token.get('refresh_token'),
                token_type=token.get('token_type', 'Bearer'),
                expires_at=datetime.utcnow() + timedelta(
                    seconds=token.get('expires_in', 32400)  # 9 hours default
                ),
                id_token=token.get('id_token'),
            )

            # Get user info
            user = await self.get_user_info(token_info.access_token)
            if not user:
                user = UserInfo(
                    id="unknown",
                    email="user@corp.local",
                    name="Corporate User"
                )

            session = self._create_session(user, token_info)
            return AuthResult.authenticated(session)

        except Exception as e:
            logger.exception("PyAuth simple token exchange failed")
            return AuthResult.failure(
                error="token_exchange_error",
                description=str(e)
            )

    async def refresh_token(self, session: AuthSession) -> AuthResult:
        """Refresh token using requests-oauthlib"""
        if not session.token or not session.token.refresh_token:
            return AuthResult.failure(
                error="no_refresh_token",
                description="No refresh token available"
            )

        try:
            oauth = self._get_oauth_session()

            token = oauth.refresh_token(
                self.config.token_endpoint,
                refresh_token=session.token.refresh_token,
                client_id=self.config.client_id,
                client_secret=self.config.client_secret,
            )

            new_token = TokenInfo(
                access_token=token['access_token'],
                refresh_token=token.get('refresh_token', session.token.refresh_token),
                token_type=token.get('token_type', 'Bearer'),
                expires_at=datetime.utcnow() + timedelta(
                    seconds=token.get('expires_in', 32400)
                ),
            )

            session.token = new_token
            return AuthResult.authenticated(session)

        except Exception as e:
            logger.exception("PyAuth simple token refresh failed")
            return AuthResult.failure(
                error="refresh_error",
                description=str(e)
            )

    async def get_user_info(self, access_token: str) -> Optional[UserInfo]:
        """Get user info from configured endpoint"""
        if not self.config.userinfo_endpoint:
            return None

        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    self.config.userinfo_endpoint,
                    headers={"Authorization": f"Bearer {access_token}"}
                )

                if response.status_code != 200:
                    return None

                claims = response.json()
                return UserInfo.from_claims(claims)

        except Exception as e:
            logger.exception("PyAuth userinfo request failed")
            return None

    async def logout(self, session: AuthSession) -> Optional[str]:
        """Return logout redirect URL"""
        return self.config.post_logout_redirect_uri

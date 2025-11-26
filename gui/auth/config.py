"""
Authentication Configuration

Loads authentication settings from environment variables.
Supports multiple providers: okta, azure, pyauth, disabled
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class AuthProvider(Enum):
    """Supported authentication providers"""
    DISABLED = "disabled"
    OKTA = "okta"
    AZURE = "azure"
    PYAUTH = "pyauth"  # Your internal PyAuthClient


class SessionStoreType(Enum):
    """Session storage backends"""
    MEMORY = "memory"
    REDIS = "redis"
    ORACLE = "oracle"


@dataclass
class OktaConfig:
    """Okta OIDC configuration"""
    issuer_url: str = ""
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = ""
    scopes: List[str] = field(default_factory=lambda: ["openid", "profile", "email"])

    @classmethod
    def from_env(cls) -> "OktaConfig":
        scopes_str = os.getenv("OKTA_SCOPES", "openid profile email")
        return cls(
            issuer_url=os.getenv("OKTA_ISSUER_URL", ""),
            client_id=os.getenv("OKTA_CLIENT_ID", ""),
            client_secret=os.getenv("OKTA_CLIENT_SECRET", ""),
            redirect_uri=os.getenv("OKTA_REDIRECT_URI", "http://localhost:8080/auth/callback"),
            scopes=scopes_str.split(),
        )

    def is_valid(self) -> bool:
        return bool(self.issuer_url and self.client_id and self.client_secret)


@dataclass
class AzureConfig:
    """Azure AD OIDC configuration"""
    tenant_id: str = ""
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = ""
    scopes: List[str] = field(default_factory=lambda: ["openid", "profile", "email"])

    @classmethod
    def from_env(cls) -> "AzureConfig":
        scopes_str = os.getenv("AZURE_SCOPES", "openid profile email")
        return cls(
            tenant_id=os.getenv("AZURE_TENANT_ID", ""),
            client_id=os.getenv("AZURE_CLIENT_ID", ""),
            client_secret=os.getenv("AZURE_CLIENT_SECRET", ""),
            redirect_uri=os.getenv("AZURE_REDIRECT_URI", "http://localhost:8080/auth/callback"),
            scopes=scopes_str.split(),
        )

    def is_valid(self) -> bool:
        return bool(self.tenant_id and self.client_id and self.client_secret)


@dataclass
class PyAuthConfig:
    """
    PyAuth Client configuration (your internal module).

    PyAuthClient extends requests-oauthlib OAuth2Session.
    These settings are used to instantiate PyAuthClient.
    """
    # Core endpoints
    secure_endpoint: str = ""
    access_url: str = ""

    # Application identity
    cons_app_id: str = ""
    client_id: str = ""
    client_secret: str = ""

    # User credentials (for grant_type=password)
    username: str = ""
    password: str = ""

    # OAuth settings
    grant_type: str = "client_credentials"  # or "password"
    scope: str = "openid profile email"

    # Additional endpoints
    pyusage_endpoint: str = ""
    api_keys_endpoint: str = ""  # Endpoint to retrieve LLM API keys
    user_info_endpoint: str = ""  # Endpoint to get user details

    # SSL/TLS
    ca_cert_location: str = ""
    verify_ssl: bool = True

    # Proxy settings (for Azure AD via PyAuth)
    proxy_enabled: bool = False
    proxy_url: str = ""

    # Azure AD specific (when using Azure through PyAuth)
    azure_scope: str = ""

    @classmethod
    def from_env(cls) -> "PyAuthConfig":
        return cls(
            secure_endpoint=os.getenv("PYAUTH_SECURE_ENDPOINT", ""),
            access_url=os.getenv("PYAUTH_CLIENT_ACCESS_URL", ""),
            cons_app_id=os.getenv("PYAUTH_CONS_APP_ID", ""),
            client_id=os.getenv("PYAUTH_CLIENT_CLIENT_ID", ""),
            client_secret=os.getenv("PYAUTH_CLIENT_CLIENT_SECRET", ""),
            username=os.getenv("PYAUTH_CLIENT_USERNAME", ""),
            password=os.getenv("PYAUTH_CLIENT_PASSWORD", ""),
            grant_type=os.getenv("PYAUTH_GRANT_TYPE", "client_credentials"),
            scope=os.getenv("PYAUTH_SCOPE", "openid profile email"),
            pyusage_endpoint=os.getenv("PYAUTH_PYUSAGE_ENDPOINT", ""),
            api_keys_endpoint=os.getenv("PYAUTH_API_KEYS_ENDPOINT", ""),
            user_info_endpoint=os.getenv("PYAUTH_USER_INFO_ENDPOINT", ""),
            ca_cert_location=os.getenv("PYAUTH_CA_CERT_LOCATION", ""),
            verify_ssl=os.getenv("PYAUTH_VERIFY_SSL", "true").lower() == "true",
            proxy_enabled=os.getenv("PYAUTH_PROXY_ENABLED", "false").lower() == "true",
            proxy_url=os.getenv("PYAUTH_PROXY_URL", ""),
            azure_scope=os.getenv("PYAUTH_AZURE_SCOPE", ""),
        )

    def is_valid(self) -> bool:
        # Minimum required: secure_endpoint and some form of credentials
        has_endpoint = bool(self.secure_endpoint)
        has_client_creds = bool(self.client_id and self.client_secret)
        has_user_creds = bool(self.username and self.password)
        return has_endpoint and (has_client_creds or has_user_creds)


@dataclass
class SessionConfig:
    """Session storage configuration"""
    store_type: SessionStoreType = SessionStoreType.MEMORY

    # Session settings
    timeout_hours: int = 9  # Your corporate standard
    cookie_name: str = "paraphrase_me_session"
    cookie_secure: bool = True  # Set to False for local development
    cookie_httponly: bool = True

    # Redis settings
    redis_url: str = ""

    # Oracle settings
    oracle_dsn: str = ""
    oracle_table: str = "auth_sessions"

    @classmethod
    def from_env(cls) -> "SessionConfig":
        store_type_str = os.getenv("SESSION_STORE", "memory").lower()
        try:
            store_type = SessionStoreType(store_type_str)
        except ValueError:
            store_type = SessionStoreType.MEMORY

        return cls(
            store_type=store_type,
            timeout_hours=int(os.getenv("SESSION_TIMEOUT_HOURS", "9")),
            cookie_name=os.getenv("SESSION_COOKIE_NAME", "paraphrase_me_session"),
            cookie_secure=os.getenv("SESSION_COOKIE_SECURE", "true").lower() == "true",
            cookie_httponly=os.getenv("SESSION_COOKIE_HTTPONLY", "true").lower() == "true",
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            oracle_dsn=os.getenv("ORACLE_DSN", ""),
            oracle_table=os.getenv("ORACLE_SESSION_TABLE", "auth_sessions"),
        )


@dataclass
class AuthConfig:
    """
    Main authentication configuration.

    Aggregates all provider-specific configs and session settings.
    """
    enabled: bool = False
    provider: AuthProvider = AuthProvider.DISABLED

    # Provider-specific configs
    okta: OktaConfig = field(default_factory=OktaConfig)
    azure: AzureConfig = field(default_factory=AzureConfig)
    pyauth: PyAuthConfig = field(default_factory=PyAuthConfig)

    # Session settings
    session: SessionConfig = field(default_factory=SessionConfig)

    # Application settings
    app_name: str = "Paraphrase Me"
    login_redirect_path: str = "/login"
    after_login_path: str = "/"
    after_logout_path: str = "/login"

    @classmethod
    def from_env(cls) -> "AuthConfig":
        """Load configuration from environment variables"""
        enabled = os.getenv("AUTH_ENABLED", "false").lower() == "true"

        provider_str = os.getenv("AUTH_PROVIDER", "disabled").lower()
        try:
            provider = AuthProvider(provider_str)
        except ValueError:
            provider = AuthProvider.DISABLED

        return cls(
            enabled=enabled,
            provider=provider,
            okta=OktaConfig.from_env(),
            azure=AzureConfig.from_env(),
            pyauth=PyAuthConfig.from_env(),
            session=SessionConfig.from_env(),
            app_name=os.getenv("AUTH_APP_NAME", "Paraphrase Me"),
            login_redirect_path=os.getenv("AUTH_LOGIN_PATH", "/login"),
            after_login_path=os.getenv("AUTH_AFTER_LOGIN_PATH", "/"),
            after_logout_path=os.getenv("AUTH_AFTER_LOGOUT_PATH", "/login"),
        )

    def get_active_provider_config(self):
        """Return the configuration for the active provider"""
        if self.provider == AuthProvider.OKTA:
            return self.okta
        elif self.provider == AuthProvider.AZURE:
            return self.azure
        elif self.provider == AuthProvider.PYAUTH:
            return self.pyauth
        return None

    def validate(self) -> tuple[bool, str]:
        """Validate that the configuration is complete for the selected provider"""
        if not self.enabled:
            return True, "Authentication disabled"

        if self.provider == AuthProvider.DISABLED:
            return True, "Authentication disabled"

        config = self.get_active_provider_config()
        if config is None:
            return False, f"Unknown provider: {self.provider}"

        if not config.is_valid():
            return False, f"Incomplete configuration for provider: {self.provider.value}"

        return True, "Configuration valid"


# Global config instance (loaded once at startup)
_config: Optional[AuthConfig] = None


def get_auth_config() -> AuthConfig:
    """Get the global auth configuration (singleton)"""
    global _config
    if _config is None:
        _config = AuthConfig.from_env()
    return _config


def reload_auth_config() -> AuthConfig:
    """Reload configuration from environment (useful for testing)"""
    global _config
    _config = AuthConfig.from_env()
    return _config

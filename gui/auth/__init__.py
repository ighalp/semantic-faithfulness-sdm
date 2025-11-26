"""
Authentication Module for Paraphrase Me GUI

==============================================================================
EXPERIMENTAL - ENTERPRISE ONLY
==============================================================================
This authentication module is currently in DEVELOPMENT and has NOT been fully
tested in production environments. It may require debugging and tuning when
deployed.

This feature is ONLY needed for enterprise deployments that require corporate
SSO integration. For personal or development use, leave authentication
disabled (the default setting: AUTH_PROVIDER=disabled or not set).
==============================================================================

Provides pluggable OAuth2/OIDC authentication with support for:
- Okta
- Azure AD
- PyAuth (custom internal provider based on requests-oauthlib)

Usage:
    from gui.auth import AuthManager, require_auth, register_auth_pages

    # In app.py startup
    auth_manager = get_auth_manager()
    register_auth_pages(base_url="http://localhost:8080")

    # Protect routes
    @ui.page('/input')
    @require_auth()
    def input_page():
        ...

Configuration (environment variables):
    AUTH_PROVIDER: disabled|okta|azure|pyauth (default: disabled)
    AUTH_SESSION_STORE: memory|redis|oracle
    AUTH_SESSION_TIMEOUT_HOURS: 9 (default)

    # Okta
    OKTA_ISSUER_URL: https://your-org.okta.com/oauth2/default
    OKTA_CLIENT_ID: your-client-id
    OKTA_CLIENT_SECRET: your-client-secret

    # Azure AD
    AZURE_TENANT_ID: your-tenant-id
    AZURE_CLIENT_ID: your-client-id
    AZURE_CLIENT_SECRET: your-client-secret

    # PyAuth (internal)
    PYAUTH_*: See config.py for full list
"""

from .manager import AuthManager, get_auth_manager, reset_auth_manager
from .decorators import (
    require_auth,
    require_role,
    inject_api_keys,
    get_current_session,
    get_current_user,
    get_session_id,
    set_session_cookie,
    clear_session_cookie,
    AuthContext,
)
from .models import UserInfo, AuthResult, AuthSession, TokenInfo, ApiKeys
from .config import AuthConfig, AuthProvider, SessionStoreType, get_auth_config
from .pages import register_auth_pages, create_user_menu, create_login_card
from .session_store import SessionStore, create_session_store

__all__ = [
    # Manager
    'AuthManager',
    'get_auth_manager',
    'reset_auth_manager',
    # Decorators
    'require_auth',
    'require_role',
    'inject_api_keys',
    'get_current_session',
    'get_current_user',
    'get_session_id',
    'set_session_cookie',
    'clear_session_cookie',
    'AuthContext',
    # Models
    'UserInfo',
    'AuthResult',
    'AuthSession',
    'TokenInfo',
    'ApiKeys',
    # Config
    'AuthConfig',
    'AuthProvider',
    'SessionStoreType',
    'get_auth_config',
    # Pages
    'register_auth_pages',
    'create_user_menu',
    'create_login_card',
    # Session Store
    'SessionStore',
    'create_session_store',
]

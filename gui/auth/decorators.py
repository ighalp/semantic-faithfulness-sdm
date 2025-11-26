"""
Authentication Decorators

Provides decorators for protecting NiceGUI pages and routes.
"""

import functools
import logging
from typing import Callable, Optional, Any

from nicegui import app, ui

from .manager import get_auth_manager
from .models import AuthSession

logger = logging.getLogger(__name__)

# Session cookie name
SESSION_COOKIE_NAME = "paraphrase_me_session"


def get_session_id() -> Optional[str]:
    """Get session ID from request cookie"""
    try:
        return app.storage.browser.get(SESSION_COOKIE_NAME)
    except Exception:
        return None


def set_session_cookie(session_id: str) -> None:
    """Set session cookie in browser storage"""
    try:
        app.storage.browser[SESSION_COOKIE_NAME] = session_id
    except Exception as e:
        logger.error(f"Failed to set session cookie: {e}")


def clear_session_cookie() -> None:
    """Clear session cookie"""
    try:
        if SESSION_COOKIE_NAME in app.storage.browser:
            del app.storage.browser[SESSION_COOKIE_NAME]
    except Exception as e:
        logger.error(f"Failed to clear session cookie: {e}")


async def get_current_session() -> Optional[AuthSession]:
    """
    Get the current user's session.

    Returns:
        AuthSession if authenticated, None otherwise
    """
    session_id = get_session_id()
    if not session_id:
        return None

    auth_manager = get_auth_manager()
    return await auth_manager.validate_session(session_id)


async def get_current_user() -> Optional[dict]:
    """
    Get current authenticated user info.

    Returns dict with user details or None if not authenticated.
    """
    session = await get_current_session()
    if session and session.user:
        return {
            "id": session.user.id,
            "email": session.user.email,
            "name": session.user.name,
            "given_name": session.user.given_name,
            "family_name": session.user.family_name,
            "picture": session.user.picture,
            "organization": session.user.organization,
        }
    return None


def require_auth(
    redirect_to: str = "/login",
    message: Optional[str] = None,
) -> Callable:
    """
    Decorator to require authentication for a page.

    Usage:
        @ui.page('/protected')
        @require_auth()
        def protected_page():
            ui.label('Secret content')

    Args:
        redirect_to: Where to redirect unauthenticated users
        message: Optional message to show on redirect

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            auth_manager = get_auth_manager()

            # Skip auth check if disabled
            if not auth_manager.is_enabled:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)

            # Check for valid session
            session = await get_current_session()

            if not session:
                logger.debug(f"Unauthenticated access to {func.__name__}, redirecting")

                # Store intended destination
                try:
                    # Get current path to redirect back after login
                    from nicegui import context
                    intended_path = context.client.page.path
                    app.storage.browser["auth_redirect"] = intended_path
                except Exception:
                    pass

                # Show message if provided
                if message:
                    ui.notify(message, type="warning")

                # Redirect to login
                ui.navigate.to(redirect_to)
                return

            # User is authenticated - inject session into kwargs if requested
            if "session" in func.__code__.co_varnames:
                kwargs["session"] = session

            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper
    return decorator


def require_role(
    roles: list[str],
    redirect_to: str = "/unauthorized",
) -> Callable:
    """
    Decorator to require specific roles.

    Usage:
        @ui.page('/admin')
        @require_auth()
        @require_role(['admin', 'superuser'])
        def admin_page():
            ui.label('Admin content')

    Args:
        roles: List of allowed roles
        redirect_to: Where to redirect unauthorized users
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            session = await get_current_session()

            if not session:
                ui.navigate.to("/login")
                return

            # Check roles from user claims
            user_roles = session.user.raw_claims.get("roles", [])
            if isinstance(user_roles, str):
                user_roles = [user_roles]

            if not any(role in user_roles for role in roles):
                logger.warning(
                    f"User {session.user.email} lacks required roles {roles}"
                )
                ui.navigate.to(redirect_to)
                return

            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper
    return decorator


# Import asyncio for coroutine checks
import asyncio


class AuthContext:
    """
    Context manager for auth-related operations in pages.

    Usage:
        async with AuthContext() as auth:
            if auth.is_authenticated:
                ui.label(f'Welcome, {auth.user.name}')
            else:
                ui.label('Please log in')
    """

    def __init__(self):
        self.session: Optional[AuthSession] = None
        self._auth_manager = get_auth_manager()

    @property
    def is_authenticated(self) -> bool:
        return self.session is not None

    @property
    def is_enabled(self) -> bool:
        return self._auth_manager.is_enabled

    @property
    def user(self):
        return self.session.user if self.session else None

    @property
    def api_keys(self):
        return self.session.api_keys if self.session else None

    async def __aenter__(self):
        self.session = await get_current_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def logout(self) -> Optional[str]:
        """Log out and return redirect URL"""
        if self.session:
            logout_url = await self._auth_manager.logout(self.session.session_id)
            clear_session_cookie()
            return logout_url
        return None


def inject_api_keys() -> Callable:
    """
    Decorator to inject API keys from session into environment.

    This allows existing code that reads from os.environ to work
    transparently with session-provided API keys.

    Usage:
        @ui.page('/analyze')
        @require_auth()
        @inject_api_keys()
        def analyze_page():
            # os.environ now has API keys from session
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            import os

            session = await get_current_session()
            if session and session.api_keys:
                keys = session.api_keys

                # Temporarily set environment variables
                original_env = {}

                if keys.openai:
                    original_env["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
                    os.environ["OPENAI_API_KEY"] = keys.openai

                if keys.anthropic:
                    original_env["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY")
                    os.environ["ANTHROPIC_API_KEY"] = keys.anthropic

                if keys.gemini:
                    original_env["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
                    os.environ["GOOGLE_API_KEY"] = keys.gemini

                # Set any extra keys
                for key, value in keys.extra.items():
                    env_key = f"{key.upper()}_API_KEY"
                    original_env[env_key] = os.environ.get(env_key)
                    os.environ[env_key] = value

                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    return func(*args, **kwargs)
                finally:
                    # Restore original environment
                    for key, value in original_env.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
            else:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)

        return wrapper
    return decorator

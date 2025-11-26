"""
Authentication Pages

NiceGUI pages for login, callback, and logout flows.
"""

import logging
from typing import Optional
from urllib.parse import urlencode

from nicegui import ui, app

from .manager import get_auth_manager
from .decorators import (
    get_session_id,
    set_session_cookie,
    clear_session_cookie,
    get_current_session,
    SESSION_COOKIE_NAME,
)

logger = logging.getLogger(__name__)


def register_auth_pages(base_url: str = "http://localhost:8080") -> None:
    """
    Register authentication pages with NiceGUI.

    Call this at application startup.

    Args:
        base_url: Base URL of the application (for redirect URIs)
    """

    @ui.page("/login")
    async def login_page():
        """Login page - initiates OAuth flow"""
        auth_manager = get_auth_manager()

        # Check if already authenticated
        session = await get_current_session()
        if session:
            # Already logged in, redirect to home or intended destination
            redirect_to = app.storage.browser.get("auth_redirect", "/")
            app.storage.browser.pop("auth_redirect", None)
            ui.navigate.to(redirect_to)
            return

        # Check if auth is enabled
        if not auth_manager.is_enabled:
            ui.navigate.to("/")
            return

        # Build callback URL
        callback_url = f"{base_url}/auth/callback"

        # Start login flow
        result = await auth_manager.start_login(callback_url)

        if result.success and result.redirect_url:
            # Redirect to IdP
            ui.navigate.to(result.redirect_url, new_tab=False)
        else:
            # Show error page
            with ui.column().classes("w-full max-w-md mx-auto mt-20 p-8"):
                ui.label("Authentication Error").classes("text-2xl font-bold text-red-600")
                ui.label(result.error or "Unknown error").classes("text-gray-600 mt-2")
                if result.error_description:
                    ui.label(result.error_description).classes("text-sm text-gray-500 mt-1")
                ui.button("Try Again", on_click=lambda: ui.navigate.to("/login")).classes("mt-4")

    @ui.page("/auth/callback")
    async def auth_callback():
        """OAuth callback handler"""
        auth_manager = get_auth_manager()

        # Get query parameters
        from nicegui import context
        request = context.client.request

        code = request.query_params.get("code")
        state = request.query_params.get("state")
        error = request.query_params.get("error")
        error_description = request.query_params.get("error_description")

        # Handle IdP errors
        if error:
            logger.warning(f"OAuth error from IdP: {error} - {error_description}")
            with ui.column().classes("w-full max-w-md mx-auto mt-20 p-8"):
                ui.label("Authentication Failed").classes("text-2xl font-bold text-red-600")
                ui.label(error).classes("text-gray-600 mt-2")
                if error_description:
                    ui.label(error_description).classes("text-sm text-gray-500 mt-1")
                ui.button("Try Again", on_click=lambda: ui.navigate.to("/login")).classes("mt-4")
            return

        # Validate required params
        if not code or not state:
            logger.warning("Missing code or state in callback")
            with ui.column().classes("w-full max-w-md mx-auto mt-20 p-8"):
                ui.label("Invalid Callback").classes("text-2xl font-bold text-red-600")
                ui.label("Missing required parameters").classes("text-gray-600 mt-2")
                ui.button("Try Again", on_click=lambda: ui.navigate.to("/login")).classes("mt-4")
            return

        # Process callback
        callback_url = f"{base_url}/auth/callback"
        result = await auth_manager.handle_callback(code, state, callback_url)

        if result.success and result.session:
            # Set session cookie
            set_session_cookie(result.session.session_id)

            # Get redirect destination
            redirect_to = app.storage.browser.get("auth_redirect", "/")
            app.storage.browser.pop("auth_redirect", None)

            logger.info(f"User logged in: {result.session.user.email}")

            # Redirect to intended destination
            ui.navigate.to(redirect_to)
        else:
            # Show error
            logger.error(f"Callback failed: {result.error} - {result.error_description}")
            with ui.column().classes("w-full max-w-md mx-auto mt-20 p-8"):
                ui.label("Authentication Failed").classes("text-2xl font-bold text-red-600")
                ui.label(result.error or "Unknown error").classes("text-gray-600 mt-2")
                if result.error_description:
                    ui.label(result.error_description).classes("text-sm text-gray-500 mt-1")
                ui.button("Try Again", on_click=lambda: ui.navigate.to("/login")).classes("mt-4")

    @ui.page("/logout")
    async def logout_page():
        """Logout handler"""
        auth_manager = get_auth_manager()

        session_id = get_session_id()
        logout_url = None

        if session_id:
            logout_url = await auth_manager.logout(session_id)

        # Clear session cookie
        clear_session_cookie()

        if logout_url:
            # Redirect to IdP for single sign-out
            ui.navigate.to(logout_url, new_tab=False)
        else:
            # Show logged out page
            with ui.column().classes("w-full max-w-md mx-auto mt-20 p-8 text-center"):
                ui.label("Logged Out").classes("text-2xl font-bold")
                ui.label("You have been successfully logged out.").classes("text-gray-600 mt-2")
                ui.button("Log In Again", on_click=lambda: ui.navigate.to("/login")).classes("mt-4")

    @ui.page("/unauthorized")
    async def unauthorized_page():
        """Unauthorized access page"""
        with ui.column().classes("w-full max-w-md mx-auto mt-20 p-8 text-center"):
            ui.label("Access Denied").classes("text-2xl font-bold text-red-600")
            ui.label(
                "You do not have permission to access this resource."
            ).classes("text-gray-600 mt-2")

            with ui.row().classes("mt-4 gap-4"):
                ui.button("Go Home", on_click=lambda: ui.navigate.to("/"))
                ui.button("Log Out", on_click=lambda: ui.navigate.to("/logout"))


def create_user_menu() -> None:
    """
    Create user menu component for header.

    Shows login button if not authenticated, user info and logout if authenticated.
    Call this within a page layout.
    """
    import asyncio

    async def build_menu():
        auth_manager = get_auth_manager()

        if not auth_manager.is_enabled:
            return

        session = await get_current_session()

        with ui.row().classes("items-center gap-2"):
            if session and session.user:
                # User is logged in
                user = session.user

                # User avatar or icon
                if user.picture:
                    ui.image(user.picture).classes("w-8 h-8 rounded-full")
                else:
                    ui.icon("person").classes("text-2xl")

                # User name dropdown
                with ui.menu() as menu:
                    ui.menu_item(
                        f"Signed in as {user.email}",
                        auto_close=False
                    ).classes("font-semibold")
                    ui.separator()
                    if user.name:
                        ui.menu_item(user.name, auto_close=False)
                    if user.organization:
                        ui.menu_item(f"Org: {user.organization}", auto_close=False)
                    ui.separator()
                    ui.menu_item("Log Out", on_click=lambda: ui.navigate.to("/logout"))

                ui.button(
                    user.name or user.email.split("@")[0],
                    on_click=menu.open
                ).props("flat").classes("text-sm")

            else:
                # Not logged in
                ui.button(
                    "Log In",
                    on_click=lambda: ui.navigate.to("/login")
                ).props("flat")

    # Run async function
    asyncio.create_task(build_menu())


def create_login_card(
    title: str = "Sign In",
    subtitle: Optional[str] = None,
    show_providers: bool = True,
) -> None:
    """
    Create a login card component.

    Can be used on custom login pages.

    Args:
        title: Card title
        subtitle: Optional subtitle
        show_providers: Whether to show provider-specific login buttons
    """
    auth_manager = get_auth_manager()

    with ui.card().classes("w-full max-w-sm mx-auto p-6"):
        ui.label(title).classes("text-2xl font-bold text-center")

        if subtitle:
            ui.label(subtitle).classes("text-gray-500 text-center mt-1")

        ui.separator().classes("my-4")

        if not auth_manager.is_enabled:
            ui.label("Authentication is not configured").classes("text-gray-500 text-center")
            return

        provider = auth_manager.get_provider()
        if provider and show_providers:
            # Show provider-specific login button
            button_text = provider.display_name

            # Provider-specific styling
            button_classes = "w-full"
            icon = "login"

            if provider.name == "azure":
                icon = "business"  # Microsoft icon alternative
            elif provider.name == "okta":
                icon = "security"
            elif provider.name == "pyauth":
                icon = "corporate_fare"

            with ui.button(
                on_click=lambda: ui.navigate.to("/login")
            ).classes(button_classes):
                ui.icon(icon).classes("mr-2")
                ui.label(button_text)
        else:
            ui.button(
                "Sign In",
                on_click=lambda: ui.navigate.to("/login")
            ).classes("w-full")

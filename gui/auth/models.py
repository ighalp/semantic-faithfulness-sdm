"""
Authentication Models

Data classes for user info, sessions, and authentication results.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import uuid
import json


@dataclass
class UserInfo:
    """
    Authenticated user information.

    Populated from OAuth token claims or user info endpoint.
    """
    # Core identity
    id: str = ""  # Unique user identifier (sub claim)
    email: str = ""
    name: str = ""

    # Optional profile fields
    given_name: str = ""
    family_name: str = ""
    picture: str = ""  # Avatar URL

    # Organization info (if available)
    organization: str = ""
    department: str = ""

    # Raw claims/attributes from provider
    raw_claims: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_claims(cls, claims: Dict[str, Any]) -> "UserInfo":
        """Create UserInfo from OAuth token claims or userinfo response"""
        return cls(
            id=claims.get("sub", claims.get("oid", "")),  # sub (standard) or oid (Azure)
            email=claims.get("email", claims.get("preferred_username", "")),
            name=claims.get("name", ""),
            given_name=claims.get("given_name", ""),
            family_name=claims.get("family_name", ""),
            picture=claims.get("picture", ""),
            organization=claims.get("org", claims.get("tenant", "")),
            department=claims.get("department", ""),
            raw_claims=claims,
        )

    @property
    def display_name(self) -> str:
        """User's display name (name, email, or id)"""
        return self.name or self.email or self.id or "Unknown User"

    @property
    def initials(self) -> str:
        """User's initials for avatar display"""
        if self.given_name and self.family_name:
            return f"{self.given_name[0]}{self.family_name[0]}".upper()
        elif self.name:
            parts = self.name.split()
            if len(parts) >= 2:
                return f"{parts[0][0]}{parts[-1][0]}".upper()
            return self.name[0].upper()
        elif self.email:
            return self.email[0].upper()
        return "?"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for session storage)"""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "given_name": self.given_name,
            "family_name": self.family_name,
            "picture": self.picture,
            "organization": self.organization,
            "department": self.department,
            "raw_claims": self.raw_claims,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserInfo":
        """Deserialize from dictionary"""
        return cls(
            id=data.get("id", ""),
            email=data.get("email", ""),
            name=data.get("name", ""),
            given_name=data.get("given_name", ""),
            family_name=data.get("family_name", ""),
            picture=data.get("picture", ""),
            organization=data.get("organization", ""),
            department=data.get("department", ""),
            raw_claims=data.get("raw_claims", {}),
        )


@dataclass
class TokenInfo:
    """OAuth token information"""
    access_token: str = ""
    refresh_token: str = ""
    token_type: str = "Bearer"
    expires_at: Optional[datetime] = None
    scope: str = ""

    # Additional token data
    id_token: str = ""  # OIDC id_token (contains user claims)
    raw_token: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if token is expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() >= self.expires_at

    @property
    def expires_in_seconds(self) -> int:
        """Seconds until token expires"""
        if self.expires_at is None:
            return -1
        delta = self.expires_at - datetime.utcnow()
        return max(0, int(delta.total_seconds()))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "scope": self.scope,
            "id_token": self.id_token,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenInfo":
        """Deserialize from dictionary"""
        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"])

        return cls(
            access_token=data.get("access_token", ""),
            refresh_token=data.get("refresh_token", ""),
            token_type=data.get("token_type", "Bearer"),
            expires_at=expires_at,
            scope=data.get("scope", ""),
            id_token=data.get("id_token", ""),
        )

    @classmethod
    def from_oauth_response(cls, token_data: Dict[str, Any]) -> "TokenInfo":
        """Create from OAuth token response"""
        expires_at = None
        if "expires_at" in token_data:
            # Unix timestamp
            expires_at = datetime.utcfromtimestamp(token_data["expires_at"])
        elif "expires_in" in token_data:
            # Seconds from now
            expires_at = datetime.utcnow() + timedelta(seconds=int(token_data["expires_in"]))

        return cls(
            access_token=token_data.get("access_token", ""),
            refresh_token=token_data.get("refresh_token", ""),
            token_type=token_data.get("token_type", "Bearer"),
            expires_at=expires_at,
            scope=token_data.get("scope", ""),
            id_token=token_data.get("id_token", ""),
            raw_token=token_data,
        )


@dataclass
class ApiKeys:
    """LLM API keys retrieved via authentication"""
    openai: str = ""
    anthropic: str = ""
    gemini: str = ""

    # Additional keys can be added
    extra: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        """Serialize to dictionary"""
        result = {
            "openai": self.openai,
            "anthropic": self.anthropic,
            "gemini": self.gemini,
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApiKeys":
        """Deserialize from dictionary"""
        known_keys = {"openai", "anthropic", "gemini"}
        extra = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            openai=data.get("openai", data.get("OPENAI_API_KEY", "")),
            anthropic=data.get("anthropic", data.get("ANTHROPIC_API_KEY", "")),
            gemini=data.get("gemini", data.get("GEMINI_API_KEY", data.get("GOOGLE_API_KEY", ""))),
            extra=extra,
        )

    def has_any(self) -> bool:
        """Check if any API keys are available"""
        return bool(self.openai or self.anthropic or self.gemini or self.extra)


@dataclass
class AuthSession:
    """
    User authentication session.

    Stored in session store (memory, Redis, or Oracle).
    """
    # Session identity
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # User info
    user: Optional[UserInfo] = None

    # Token info
    token: Optional[TokenInfo] = None

    # API keys (retrieved via auth)
    api_keys: Optional[ApiKeys] = None

    # Session metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    # Provider that created this session
    provider: str = ""

    @property
    def is_valid(self) -> bool:
        """Check if session is still valid"""
        if self.expires_at and datetime.utcnow() >= self.expires_at:
            return False
        if self.token and self.token.is_expired:
            return False
        return self.user is not None

    @property
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return self.is_valid and self.user is not None

    def touch(self):
        """Update last accessed time"""
        self.last_accessed = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for storage)"""
        return {
            "session_id": self.session_id,
            "user": self.user.to_dict() if self.user else None,
            "token": self.token.to_dict() if self.token else None,
            "api_keys": self.api_keys.to_dict() if self.api_keys else None,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "provider": self.provider,
        }

    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuthSession":
        """Deserialize from dictionary"""
        user = UserInfo.from_dict(data["user"]) if data.get("user") else None
        token = TokenInfo.from_dict(data["token"]) if data.get("token") else None
        api_keys = ApiKeys.from_dict(data["api_keys"]) if data.get("api_keys") else None

        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"])

        return cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            user=user,
            token=token,
            api_keys=api_keys,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else datetime.utcnow(),
            expires_at=expires_at,
            provider=data.get("provider", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AuthSession":
        """Deserialize from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class AuthResult:
    """Result of an authentication attempt"""
    success: bool = False
    session: Optional[AuthSession] = None
    error: str = ""
    error_description: str = ""

    # Redirect URL (for OAuth flows)
    redirect_url: str = ""

    @classmethod
    def failure(cls, error: str, description: str = "") -> "AuthResult":
        """Create a failure result"""
        return cls(success=False, error=error, error_description=description)

    @classmethod
    def redirect(cls, url: str) -> "AuthResult":
        """Create a redirect result (for OAuth authorization)"""
        return cls(success=False, redirect_url=url)

    @classmethod
    def authenticated(cls, session: AuthSession) -> "AuthResult":
        """Create a successful authentication result"""
        return cls(success=True, session=session)

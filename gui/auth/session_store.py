"""
Session Storage Backends

Provides pluggable session storage: Memory, Redis, Oracle.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict
from datetime import datetime, timedelta
import json
import logging

from .models import AuthSession
from .config import SessionConfig, SessionStoreType

logger = logging.getLogger(__name__)


class SessionStore(ABC):
    """Abstract base class for session storage"""

    @abstractmethod
    async def get(self, session_id: str) -> Optional[AuthSession]:
        """Retrieve a session by ID"""
        pass

    @abstractmethod
    async def set(self, session: AuthSession) -> None:
        """Store a session"""
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Delete a session"""
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Remove expired sessions, return count removed"""
        pass


class MemorySessionStore(SessionStore):
    """
    In-memory session storage.

    Suitable for development and single-instance deployments.
    Sessions are lost on restart.
    """

    def __init__(self):
        self._sessions: Dict[str, AuthSession] = {}

    async def get(self, session_id: str) -> Optional[AuthSession]:
        session = self._sessions.get(session_id)
        if session and session.is_valid:
            session.touch()
            return session
        elif session:
            # Expired - clean up
            del self._sessions[session_id]
        return None

    async def set(self, session: AuthSession) -> None:
        self._sessions[session.session_id] = session

    async def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def cleanup_expired(self) -> int:
        now = datetime.utcnow()
        expired = [
            sid for sid, session in self._sessions.items()
            if session.expires_at and session.expires_at <= now
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)


class RedisSessionStore(SessionStore):
    """
    Redis-backed session storage.

    Suitable for production with multiple instances.
    Requires redis package: pip install redis
    """

    def __init__(self, config: SessionConfig):
        self.config = config
        self._redis = None
        self._prefix = "paraphrase_me:session:"

    def _get_redis(self):
        """Lazy initialization of Redis client"""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(
                    self.config.redis_url,
                    decode_responses=True
                )
            except ImportError:
                raise ImportError(
                    "Redis support requires the 'redis' package. "
                    "Install with: pip install redis"
                )
        return self._redis

    def _key(self, session_id: str) -> str:
        """Generate Redis key for session"""
        return f"{self._prefix}{session_id}"

    async def get(self, session_id: str) -> Optional[AuthSession]:
        redis = self._get_redis()
        try:
            data = await redis.get(self._key(session_id))
            if data:
                session = AuthSession.from_json(data)
                if session.is_valid:
                    session.touch()
                    # Update last_accessed in Redis
                    await self.set(session)
                    return session
                else:
                    # Expired
                    await self.delete(session_id)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set(self, session: AuthSession) -> None:
        redis = self._get_redis()
        try:
            # Calculate TTL from session expiry
            ttl = self.config.timeout_hours * 3600
            if session.expires_at:
                ttl = int((session.expires_at - datetime.utcnow()).total_seconds())
                ttl = max(ttl, 60)  # Minimum 1 minute

            await redis.setex(
                self._key(session.session_id),
                ttl,
                session.to_json()
            )
        except Exception as e:
            logger.error(f"Redis set error: {e}")

    async def delete(self, session_id: str) -> None:
        redis = self._get_redis()
        try:
            await redis.delete(self._key(session_id))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")

    async def cleanup_expired(self) -> int:
        # Redis handles expiry automatically via TTL
        return 0


class OracleSessionStore(SessionStore):
    """
    Oracle database session storage.

    Suitable for enterprise deployments using Oracle.
    Requires cx_Oracle or oracledb package.

    Table schema (create manually or via migration):
        CREATE TABLE auth_sessions (
            session_id VARCHAR2(64) PRIMARY KEY,
            session_data CLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX idx_sessions_expires ON auth_sessions(expires_at);
    """

    def __init__(self, config: SessionConfig):
        self.config = config
        self._pool = None

    def _get_pool(self):
        """Lazy initialization of Oracle connection pool"""
        if self._pool is None:
            try:
                # Try oracledb first (newer, pure Python)
                try:
                    import oracledb
                    self._pool = oracledb.create_pool(
                        dsn=self.config.oracle_dsn,
                        min=1,
                        max=5,
                        increment=1
                    )
                except ImportError:
                    # Fall back to cx_Oracle
                    import cx_Oracle
                    self._pool = cx_Oracle.SessionPool(
                        dsn=self.config.oracle_dsn,
                        min=1,
                        max=5,
                        increment=1
                    )
            except ImportError:
                raise ImportError(
                    "Oracle support requires 'oracledb' or 'cx_Oracle' package. "
                    "Install with: pip install oracledb"
                )
        return self._pool

    async def get(self, session_id: str) -> Optional[AuthSession]:
        pool = self._get_pool()
        try:
            with pool.acquire() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        f"""
                        SELECT session_data FROM {self.config.oracle_table}
                        WHERE session_id = :1 AND expires_at > CURRENT_TIMESTAMP
                        """,
                        [session_id]
                    )
                    row = cursor.fetchone()
                    if row:
                        session = AuthSession.from_json(row[0])
                        session.touch()
                        # Update last_accessed
                        cursor.execute(
                            f"""
                            UPDATE {self.config.oracle_table}
                            SET last_accessed = CURRENT_TIMESTAMP,
                                session_data = :1
                            WHERE session_id = :2
                            """,
                            [session.to_json(), session_id]
                        )
                        conn.commit()
                        return session
            return None
        except Exception as e:
            logger.error(f"Oracle get error: {e}")
            return None

    async def set(self, session: AuthSession) -> None:
        pool = self._get_pool()
        try:
            with pool.acquire() as conn:
                with conn.cursor() as cursor:
                    # Upsert (merge)
                    cursor.execute(
                        f"""
                        MERGE INTO {self.config.oracle_table} t
                        USING (SELECT :1 AS session_id FROM dual) s
                        ON (t.session_id = s.session_id)
                        WHEN MATCHED THEN
                            UPDATE SET
                                session_data = :2,
                                expires_at = :3,
                                last_accessed = CURRENT_TIMESTAMP
                        WHEN NOT MATCHED THEN
                            INSERT (session_id, session_data, expires_at, last_accessed)
                            VALUES (:1, :2, :3, CURRENT_TIMESTAMP)
                        """,
                        [
                            session.session_id,
                            session.to_json(),
                            session.expires_at
                        ]
                    )
                    conn.commit()
        except Exception as e:
            logger.error(f"Oracle set error: {e}")

    async def delete(self, session_id: str) -> None:
        pool = self._get_pool()
        try:
            with pool.acquire() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        f"DELETE FROM {self.config.oracle_table} WHERE session_id = :1",
                        [session_id]
                    )
                    conn.commit()
        except Exception as e:
            logger.error(f"Oracle delete error: {e}")

    async def cleanup_expired(self) -> int:
        pool = self._get_pool()
        try:
            with pool.acquire() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        f"""
                        DELETE FROM {self.config.oracle_table}
                        WHERE expires_at <= CURRENT_TIMESTAMP
                        """
                    )
                    count = cursor.rowcount
                    conn.commit()
                    return count
        except Exception as e:
            logger.error(f"Oracle cleanup error: {e}")
            return 0


def create_session_store(config: SessionConfig) -> SessionStore:
    """
    Factory function to create the appropriate session store.

    Args:
        config: Session configuration

    Returns:
        SessionStore instance
    """
    if config.store_type == SessionStoreType.REDIS:
        return RedisSessionStore(config)
    elif config.store_type == SessionStoreType.ORACLE:
        return OracleSessionStore(config)
    else:
        return MemorySessionStore()

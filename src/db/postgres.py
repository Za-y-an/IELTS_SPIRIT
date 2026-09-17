"""
PostgreSQL Persistence Layer via SQLAlchemy 2.0 (Async) + asyncpg.

================================================================================
SUPABASE CONNECTION STRING CONVERSION GUIDE:
================================================================================
Supabase dashboard gives you a sync-style connection string under Project Settings -> Database:
  e.g.: postgresql://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres

To connect via asyncpg and SQLAlchemy 2.0 async engine, you MUST convert it:
  1. Scheme: Change 'postgresql://' to 'postgresql+asyncpg://'
  2. Percent-encode password: If your password has special characters like '@', '#', '=', or ':',
     they MUST be URL-encoded (e.g. '@' -> '%40', '#' -> '%23', '=' -> '%3D').
  3. SSL Parameter: Supabase requires SSL encryption. Append '?ssl=require' to the URI.
  4. Explicit connect_args: With asyncpg, pass connect_args={"ssl": "require"} to create_async_engine
     to ensure asyncpg enables SSL negotiation directly.
================================================================================
"""

from datetime import datetime, timezone
import traceback
from typing import List, Optional, Tuple
from sqlalchemy import BigInteger, DateTime, String, Text, func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class MessageHistory(Base):
    """WhatsApp conversation message history table for IELTS coaching clients."""

    __tablename__ = "message_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    phone_number: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # 'user', 'assistant', 'system'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<MessageHistory(id={self.id}, phone='{self.phone_number}', role='{self.role}', content='{self.content[:40]}...', created_at='{self.created_at}')>"


def get_async_engine(db_url: str) -> AsyncEngine:
    """
    Create and return an async SQLAlchemy engine for PostgreSQL using asyncpg.
    Explicitly passes connect_args={"ssl": "require"} for secure Supabase connectivity.
    """
    if not db_url or not db_url.strip():
        raise ValueError("SUPABASE_DB_URL is not set or empty.")

    cleaned_url = db_url.strip()
    if not cleaned_url.startswith("postgresql+asyncpg://"):
        if cleaned_url.startswith("postgresql://"):
            cleaned_url = cleaned_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif cleaned_url.startswith("postgres://"):
            cleaned_url = cleaned_url.replace("postgres://", "postgresql+asyncpg://", 1)

    connect_args = {}
    if "ssl=require" in cleaned_url or "pooler.supabase.com" in cleaned_url or "supabase.co" in cleaned_url:
        connect_args["ssl"] = "require"
    if "pooler.supabase.com" in cleaned_url or ":6543" in cleaned_url:
        # Supabase transaction pooler (port 6543 / pgbouncer) requires disabling prepared statement cache
        connect_args["statement_cache_size"] = 0

    return create_async_engine(
        cleaned_url,
        connect_args=connect_args,
        echo=False,
        future=True,
        pool_pre_ping=True,
    )


def get_async_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return an async session factory bound to the given engine."""
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )


async def verify_postgres_read_write(engine: AsyncEngine) -> Tuple[bool, str, Optional[MessageHistory], Optional[str]]:
    """
    Verify PostgreSQL / Supabase async database:
    1. Connects to the database.
    2. Verifies / creates 'message_history' table.
    3. Inserts one dummy row into 'message_history'.
    4. Reads back the row to prove read/write functionality.
    
    Returns (success: bool, status_message: str, row: Optional[MessageHistory], full_traceback: Optional[str]).
    """
    try:
        # Step 1: Create schema / table
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Step 2: Insert dummy row
        session_factory = get_async_session_factory(engine)
        async with session_factory() as session:
            test_msg = MessageHistory(
                phone_number="+10000000000",
                role="system",
                content="Verification ping: read/write check successful.",
            )
            session.add(test_msg)
            await session.commit()
            await session.refresh(test_msg)
            inserted_id = test_msg.id

            # Step 3: Read back the row
            stmt = select(MessageHistory).where(MessageHistory.id == inserted_id)
            result = await session.execute(stmt)
            fetched_row = result.scalar_one_or_none()

            if fetched_row is None:
                return False, "Failed to read back the inserted dummy row.", None, None

            return True, f"Verified: 'message_history' table exists; inserted & read back row ID={fetched_row.id}.", fetched_row, None

    except Exception as exc:
        full_tb = traceback.format_exc()
        return False, f"PostgreSQL verification failed: {exc}", None, full_tb


async def log_rag_interaction(
    engine: AsyncEngine,
    phone_number: str,
    question: str,
    answer: str,
) -> Tuple[bool, List[MessageHistory], Optional[str]]:
    """
    Log a user question and assistant answer exchange to message_history table.
    Queries both newly inserted rows back by their primary key IDs to confirm persistence.
    Returns (success: bool, rows: List[MessageHistory], error: Optional[str]).
    """
    try:
        session_factory = get_async_session_factory(engine)
        async with session_factory() as session:
            user_msg = MessageHistory(
                phone_number=phone_number,
                role="user",
                content=question,
            )
            assistant_msg = MessageHistory(
                phone_number=phone_number,
                role="assistant",
                content=answer,
            )
            session.add_all([user_msg, assistant_msg])
            await session.commit()
            await session.refresh(user_msg)
            await session.refresh(assistant_msg)

            stmt = (
                select(MessageHistory)
                .where(MessageHistory.id.in_([user_msg.id, assistant_msg.id]))
                .order_by(MessageHistory.id.asc())
            )
            result = await session.execute(stmt)
            rows = list(result.scalars().all())
            return True, rows, None
    except Exception as exc:
        return False, [], str(exc)

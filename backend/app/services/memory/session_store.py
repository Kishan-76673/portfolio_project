import os
import aiosqlite
from typing import List
from app.services.llm.base import ChatMessage

class SessionMemoryManager:
    """Manages conversational session history in SQLite with sliding window."""

    def __init__(self, db_path: str = "./data/portfolio.db", window_size: int = 6):
        self.db_path = db_path
        self.window_size = window_size
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA synchronous=NORMAL;")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id);")
            await db.commit()

    async def add_message(self, session_id: str, role: str, content: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content)
            )
            await db.commit()

    async def get_history(self, session_id: str) -> List[ChatMessage]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT role, content FROM (
                    SELECT role, content, id FROM messages 
                    WHERE session_id = ? 
                    ORDER BY id DESC LIMIT ?
                ) ORDER BY id ASC
                """,
                (session_id, self.window_size)
            ) as cursor:
                rows = await cursor.fetchall()
                return [ChatMessage(role=r[0], content=r[1]) for r in rows]

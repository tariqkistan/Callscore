"""Script to create the call_scores table in SQLite."""

import asyncio

from callscore.ingest import create_tables


async def main() -> None:
    """Create database tables."""
    print("Creating database tables...")
    await create_tables()
    print("Database tables created successfully.")


if __name__ == "__main__":
    asyncio.run(main())

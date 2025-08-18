"""Database connection and session management
"""

import logging
import os

from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


logger = logging.getLogger(__name__)

# Database configuration from environment
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:password@localhost:5432/observer_insights'
)

# Engine configuration
engine_kwargs = {
    'echo': os.getenv('DATABASE_ECHO', 'false').lower() == 'true',
    'pool_size': int(os.getenv('DATABASE_POOL_SIZE', '20')),
    'max_overflow': int(os.getenv('DATABASE_MAX_OVERFLOW', '30')),
    'pool_pre_ping': True,
    'pool_recycle': 3600,  # Recycle connections every hour
}

# Handle SQLite for development/testing
if DATABASE_URL.startswith('sqlite'):
    engine_kwargs.update({
        'poolclass': StaticPool,
        'connect_args': {'check_same_thread': False}
    })
    # Remove PostgreSQL-specific options
    engine_kwargs.pop('pool_size', None)
    engine_kwargs.pop('max_overflow', None)
    engine_kwargs.pop('pool_pre_ping', None)
    engine_kwargs.pop('pool_recycle', None)

# Create engine
engine = create_engine(DATABASE_URL, **engine_kwargs)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db() -> Session:
    """Dependency to get database session
    Use with FastAPI Depends()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Database event listeners for connection management
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set SQLite pragmas for better performance and consistency"""
    if 'sqlite' in DATABASE_URL:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=1000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()


@event.listens_for(engine, "checkout")
def checkout_listener(dbapi_connection, connection_record, connection_proxy):
    """Log database connection checkout"""
    logger.debug("Database connection checked out")


@event.listens_for(engine, "checkin")
def checkin_listener(dbapi_connection, connection_record):
    """Log database connection checkin"""
    logger.debug("Database connection checked in")


class DatabaseManager:
    """Database management utilities"""

    @staticmethod
    def create_tables():
        """Create all database tables"""
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created")

    @staticmethod
    def drop_tables():
        """Drop all database tables"""
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped")

    @staticmethod
    def get_connection_info():
        """Get database connection information"""
        with engine.connect() as conn:
            if 'postgresql' in DATABASE_URL:
                result = conn.execute("SELECT version()")
                version = result.fetchone()[0]
                return {"type": "PostgreSQL", "version": version}
            elif 'sqlite' in DATABASE_URL:
                result = conn.execute("SELECT sqlite_version()")
                version = result.fetchone()[0]
                return {"type": "SQLite", "version": version}
            else:
                return {"type": "Unknown", "version": "Unknown"}

    @staticmethod
    def health_check() -> bool:
        """Check database connectivity"""
        try:
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    @staticmethod
    def get_stats():
        """Get database statistics"""
        try:
            with engine.connect() as conn:
                if 'postgresql' in DATABASE_URL:
                    # PostgreSQL specific stats
                    stats = {
                        "active_connections": conn.execute(
                            "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                        ).scalar(),
                        "database_size": conn.execute(
                            "SELECT pg_size_pretty(pg_database_size(current_database()))"
                        ).scalar(),
                    }
                elif 'sqlite' in DATABASE_URL:
                    # SQLite specific stats
                    stats = {
                        "page_count": conn.execute("PRAGMA page_count").scalar(),
                        "page_size": conn.execute("PRAGMA page_size").scalar(),
                    }
                    stats["database_size"] = f"{(stats['page_count'] * stats['page_size']) / 1024 / 1024:.2f} MB"
                else:
                    stats = {"message": "Stats not available for this database type"}

                return stats
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}


# Global database manager instance
db_manager = DatabaseManager()

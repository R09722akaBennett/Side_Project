"""
Initialize local SQLite database for development and testing
"""
import logging
import os
import sys
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db_initializer")

# Use SQLite database
SQLITE_DB = "sqlite:///linkedin_jobs.db"

def init_local_db():
    """Initialize a local SQLite database for development"""
    try:
        # Import the models but override the engine creation
        sys.path.append('.')
        from app.database.models import Base
        
        # Create engine with SQLite
        logger.info(f"Creating database with SQLite at: {SQLITE_DB}")
        engine = create_engine(SQLITE_DB)
        
        # Create all tables
        Base.metadata.create_all(engine)
        logger.info("Database initialized successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False

if __name__ == "__main__":
    if init_local_db():
        logger.info("Database initialization completed.")
    else:
        logger.error("Database initialization failed.")

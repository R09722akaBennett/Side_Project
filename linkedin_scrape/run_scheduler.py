"""
Run the LinkedIn job scheduler with local configuration overrides
"""
import os
import sys
import logging
from datetime import datetime

# Configure database to use SQLite instead of PostgreSQL
os.environ["DATABASE_URL"] = "sqlite:///linkedin_jobs.db"

# Configure OpenAI API key without proxies
os.environ["OPENAI_API_KEY"] = os.environ.get("X_AI_API_KEY", "sk-dummy-key")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("scheduler_runner")

if __name__ == "__main__":
    try:
        logger.info("Starting LinkedIn job scheduler with local configuration...")
        
        # Import after environment configuration
        from app.scheduler import start_scheduler_thread
        
        # Start the scheduler
        logger.info("Initializing scheduler thread...")
        scheduler_thread = start_scheduler_thread()
        
        logger.info("Scheduler started successfully. Press Ctrl+C to stop.")
        
        try:
            # Keep main program running
            scheduler_thread.join()
        except KeyboardInterrupt:
            logger.info("Received termination signal, stopping the application...")
            
    except Exception as e:
        logger.error(f"Error starting scheduler: {str(e)}")
        sys.exit(1)

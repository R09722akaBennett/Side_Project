"""
Temporary configuration override for local development
"""
import os

# Override database connection to use SQLite instead of PostgreSQL
os.environ["DATABASE_URL"] = "sqlite:///linkedin_jobs.db"

# X.AI API settings (using the default from config.py)
# os.environ["X_AI_API_KEY"] = "your-api-key-here"
# os.environ["X_AI_BASE_URL"] = "https://api.x.ai/v1"

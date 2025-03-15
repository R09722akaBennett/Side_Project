import os
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 資料庫配置
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "linkedin_jobs")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# 爬蟲配置
DEFAULT_KEYWORD = os.getenv("DEFAULT_KEYWORD", "AI Engineer")
DEFAULT_LOCATION = os.getenv("DEFAULT_LOCATION", "Taiwan")
DEFAULT_TIME_FILTER = os.getenv("DEFAULT_TIME_FILTER", "r86400")  # 24小時內
DEFAULT_MAX_PAGES = int(os.getenv("DEFAULT_MAX_PAGES", "5"))

# 排程配置
SCHEDULE_INTERVAL = int(os.getenv("SCHEDULE_INTERVAL", "24"))  # 每24小時執行一次 
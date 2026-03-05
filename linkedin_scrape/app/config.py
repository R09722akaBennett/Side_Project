import os
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 資料庫配置
# 先檢查是否有直接設定 DATABASE_URL 環境變數
# 如果有，則優先使用它，適用於本地開發時使用 SQLite
DATABASE_URL = os.getenv("DATABASE_URL", None) 

# 如果未設定 DATABASE_URL，則使用 PostgreSQL 連線設定
if DATABASE_URL is None:
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

# AI API 配置
# X.AI API 設置
X_AI_API_KEY = os.getenv("X_AI_API_KEY", "")
X_AI_BASE_URL = os.getenv("X_AI_BASE_URL", "https://api.x.ai/v1")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  
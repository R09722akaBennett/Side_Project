from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

from app.config import DATABASE_URL

Base = declarative_base()

class LinkedInJob(Base):
    """LinkedIn 職缺資料表模型"""
    __tablename__ = "linkedin_jobs"

    id = Column(Integer, primary_key=True)
    company_name = Column(String(255), nullable=False)
    job_title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    link = Column(String(512), nullable=False, unique=True)
    posting_date = Column(String(255), nullable=True)
    seniority_level = Column(String(255), nullable=True)
    employment_type = Column(String(255), nullable=True)
    job_function = Column(String(255), nullable=True)
    industries = Column(String(255), nullable=True)
    scrape_date = Column(DateTime, default=func.now())
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class SearchConfig(Base):
    """搜索參數配置表"""
    __tablename__ = "search_configs"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)  # 配置名稱
    keyword = Column(String(255), nullable=False)  # 搜索關鍵字
    location = Column(String(255), nullable=False)  # 搜索地點
    time_filter = Column(String(50), default='r604800')  # 時間過濾器
    max_pages = Column(Integer, default=5)  # 最大搜索頁數
    is_active = Column(Boolean, default=True)  # 是否啟用
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<SearchConfig {self.name}: {self.keyword} in {self.location}>"

def init_db():
    """初始化資料庫"""
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    return engine 
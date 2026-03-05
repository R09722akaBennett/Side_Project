from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

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
    job_scrape = Column(String(255), nullable=True)  # 搜尋關鍵字
    location = Column(String(255), nullable=True)   # 搜尋地點
    scrape_date = Column(DateTime, default=func.now())
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationship to the analyzed data
    analysis = relationship("JobAnalysis", back_populates="job", uselist=False)

class JobAnalysis(Base):
    """職缺分析資料表模型"""
    __tablename__ = "job_analysis"
    
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("linkedin_jobs.id"), nullable=False, unique=True)
    keywords = Column(Text, nullable=True)  # AI/ML/DS關鍵詞
    hard_skill = Column(Text, nullable=True)  # 硬技能
    soft_skill = Column(Text, nullable=True)  # 軟技能
    cleaned_job_title = Column(String(255), nullable=True)  # 清理後的職位名稱
    suitable_personality = Column(Text, nullable=True)  # 適合的個性特質
    representative_anime_character = Column(String(255), nullable=True)  # 代表性動漫角色
    representative_animal = Column(String(255), nullable=True)  # 代表性動物
    job_superpower = Column(String(255), nullable=True)  # 職業超能力
    job_theme_song = Column(String(255), nullable=True)  # 職業主題曲
    processed_at = Column(DateTime, default=func.now())  # 處理時間
    
    # Relationship to the job data
    job = relationship("LinkedInJob", back_populates="analysis")
    
    def __repr__(self):
        return f"<JobAnalysis {self.id}: {self.cleaned_job_title}>"

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
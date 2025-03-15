from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd

from app.database.models import LinkedInJob, init_db, SearchConfig
from app.config import DATABASE_URL

def get_db_session():
    """獲取資料庫 session"""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    return Session()

def save_jobs_to_db(jobs_df):
    """
    將職缺資料儲存到資料庫
    
    Args:
        jobs_df: 包含職缺資料的 DataFrame
    """
    session = get_db_session()
    
    try:
        for _, row in jobs_df.iterrows():
            # 檢查職缺是否已存在 (根據連結去重)
            existing_job = session.query(LinkedInJob).filter_by(link=row['Link']).first()
            
            if existing_job is None:
                job = LinkedInJob(
                    company_name=row['Company Name'],
                    job_title=row['Job Title'],
                    description=row['Description'],
                    link=row['Link'],
                    posting_date=row['Posting Date'],
                    seniority_level=row['Seniority Level'],
                    employment_type=row['Employment Type'],
                    job_function=row['Job Function'],
                    industries=row['Industries'],
                    scrape_date=datetime.now() if 'Scrape Date' not in row else row['Scrape Date']
                )
                session.add(job)
        
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"儲存資料庫時發生錯誤: {str(e)}")
        return False
    finally:
        session.close()

def get_jobs(filters=None, limit=100):
    """
    從資料庫獲取職缺資料
    
    Args:
        filters: 過濾條件字典
        limit: 最大結果數
        
    Returns:
        DataFrame: 符合條件的職缺
    """
    session = get_db_session()
    query = session.query(LinkedInJob)
    
    if filters:
        if 'company_name' in filters:
            query = query.filter(LinkedInJob.company_name.ilike(f"%{filters['company_name']}%"))
        if 'job_title' in filters:
            query = query.filter(LinkedInJob.job_title.ilike(f"%{filters['job_title']}%"))
        if 'seniority_level' in filters:
            query = query.filter(LinkedInJob.seniority_level == filters['seniority_level'])
        # 可以添加更多過濾條件
    
    jobs = query.limit(limit).all()
    
    # 轉換為 DataFrame
    job_data = []
    for job in jobs:
        job_data.append({
            'id': job.id,
            'Company Name': job.company_name,
            'Job Title': job.job_title,
            'Description': job.description,
            'Link': job.link,
            'Posting Date': job.posting_date,
            'Seniority Level': job.seniority_level,
            'Employment Type': job.employment_type,
            'Job Function': job.job_function,
            'Industries': job.industries,
            'Scrape Date': job.scrape_date,
        })
    
    session.close()
    return pd.DataFrame(job_data)

def create_search_config(name, keyword, location, time_filter='r604800', max_pages=5, is_active=True):
    """創建新的搜索配置"""
    session = get_db_session()
    try:
        config = SearchConfig(
            name=name,
            keyword=keyword,
            location=location,
            time_filter=time_filter,
            max_pages=max_pages,
            is_active=is_active
        )
        session.add(config)
        session.commit()
        return config.id
    except Exception as e:
        session.rollback()
        print(f"創建搜索配置時發生錯誤: {str(e)}")
        return None
    finally:
        session.close()

def get_all_search_configs(active_only=True):
    """獲取所有搜索配置"""
    session = get_db_session()
    try:
        query = session.query(SearchConfig)
        if active_only:
            query = query.filter(SearchConfig.is_active == True)
        configs = query.all()
        return configs
    finally:
        session.close()

def update_search_config(config_id, **kwargs):
    """更新搜索配置"""
    session = get_db_session()
    try:
        config = session.query(SearchConfig).filter_by(id=config_id).first()
        if not config:
            return False
            
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"更新搜索配置時發生錯誤: {str(e)}")
        return False
    finally:
        session.close()

def delete_search_config(config_id):
    """刪除搜索配置"""
    session = get_db_session()
    try:
        config = session.query(SearchConfig).filter_by(id=config_id).first()
        if not config:
            return False
            
        session.delete(config)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"刪除搜索配置時發生錯誤: {str(e)}")
        return False
    finally:
        session.close() 
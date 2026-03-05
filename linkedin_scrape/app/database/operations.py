from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd
from sqlalchemy import text

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
                # 處理日期格式，確保存入正確的格式
                # 由於 posting_date 在模型中定義為字串，所以保持字串格式
                # 而 scrape_date 在模型中定義為 DateTime，所以確保傳入 datetime 物件
                
                job = LinkedInJob(
                    company_name=row['Company Name'],
                    job_title=row['Job Title'],
                    description=row['Description'],
                    link=row['Link'],
                    posting_date=str(row['Posting Date']),  # 確保使用字串格式
                    seniority_level=row['Seniority Level'],
                    employment_type=row['Employment Type'],
                    job_function=row['Job Function'],
                    industries=row['Industries'],
                    job_scrape=row['Job Scrape'] if 'Job Scrape' in row else None,
                    location=row['Location'] if 'Location' in row else None,
                    # 確保使用 datetime 物件
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

def deduplicate_jobs():
    """
    去重LinkedIn職缺資料，根據company_name, job_title, description, link作為唯一鍵
    只保留每組重複資料中posting_date最新的記錄，刪除其他重複記錄
    
    Deduplicate LinkedIn job data based on company_name, job_title, description, link
    Keep only the record with the newest posting_date among duplicates, delete all others
    
    Returns:
        tuple: (刪除數量, 保留數量) - (number of deleted records, number of kept records)
    """
    session = get_db_session()
    try:
        # 使用原生SQL執行高效能的去重操作
        # Use raw SQL for high-performance deduplication
        
        # 1. 創建臨時表存儲所有重複組中posting_date最新的記錄ID
        # 1. Create a temporary table to store the IDs with the newest posting_date in each duplicate group
        create_temp_table_sql = text("""
        CREATE TEMP TABLE newest_job_ids AS
        WITH ranked_jobs AS (
            SELECT 
                id,
                company_name, 
                job_title, 
                description, 
                link,
                posting_date,
                ROW_NUMBER() OVER (
                    PARTITION BY company_name, job_title, description, link 
                    ORDER BY 
                        -- 嘗試將posting_date轉換為日期，如果失敗則使用id作為次要排序條件
                        -- Try to convert posting_date to date, if it fails use id as secondary sorting
                        CASE 
                            WHEN posting_date ~ '^\\d{4}-\\d{2}-\\d{2}' THEN posting_date::date
                            ELSE NULL
                        END DESC NULLS LAST,
                        id DESC
                ) as row_num
            FROM 
                linkedin_jobs
        )
        SELECT id as max_id
        FROM ranked_jobs
        WHERE row_num = 1
        """)
        
        # 2. 刪除所有不在臨時表中的記錄（即重複記錄中較舊的記錄）
        # 2. Delete all records not in the temporary table (older duplicates)
        delete_duplicates_sql = text("""
        DELETE FROM linkedin_jobs
        WHERE id NOT IN (SELECT max_id FROM newest_job_ids)
        RETURNING id
        """)
        
        # 執行SQL語句
        # Execute SQL statements
        session.execute(create_temp_table_sql)
        deleted_rows = session.execute(delete_duplicates_sql).fetchall()
        
        # 計算保留的記錄數
        # Calculate number of kept records
        count_kept_sql = text("SELECT COUNT(*) FROM linkedin_jobs")
        kept_count = session.execute(count_kept_sql).scalar()
        
        # 提交事務
        # Commit transaction
        session.commit()
        
        deleted_count = len(deleted_rows)
        print(f"去重完成: 已刪除 {deleted_count} 條重複記錄，保留 {kept_count} 條唯一記錄")
        print(f"Deduplication complete: Deleted {deleted_count} duplicate records, kept {kept_count} unique records")
        
        return (deleted_count, kept_count)
    
    except Exception as e:
        session.rollback()
        print(f"執行去重操作時發生錯誤: {str(e)}")
        print(f"Error during deduplication: {str(e)}")
        return (0, 0)
    finally:
        session.close() 
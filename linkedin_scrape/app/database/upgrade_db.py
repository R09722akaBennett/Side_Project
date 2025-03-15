"""
資料庫升級腳本 - 添加 job_scrape 和 location 欄位
"""

from sqlalchemy import create_engine, Column, String, text
from sqlalchemy.ext.declarative import declarative_base
from app.config import DATABASE_URL

def upgrade_database():
    """為 linkedin_jobs 表添加新欄位"""
    print("開始升級資料庫...")
    
    # 連接到資料庫
    engine = create_engine(DATABASE_URL)
    connection = engine.connect()
    
    try:
        # 檢查 job_scrape 欄位是否存在
        check_job_scrape = connection.execute(text("""
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.columns 
                WHERE table_name='linkedin_jobs' AND column_name='job_scrape'
            );
        """)).scalar()
        
        # 如果 job_scrape 欄位不存在，則添加
        if not check_job_scrape:
            print("添加 job_scrape 欄位...")
            connection.execute(text("""
                ALTER TABLE linkedin_jobs ADD COLUMN job_scrape VARCHAR(255);
            """))
        else:
            print("job_scrape 欄位已存在")
        
        # 檢查 location 欄位是否存在
        check_location = connection.execute(text("""
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.columns 
                WHERE table_name='linkedin_jobs' AND column_name='location'
            );
        """)).scalar()
        
        # 如果 location 欄位不存在，則添加
        if not check_location:
            print("添加 location 欄位...")
            connection.execute(text("""
                ALTER TABLE linkedin_jobs ADD COLUMN location VARCHAR(255);
            """))
        else:
            print("location 欄位已存在")
        
        print("資料庫升級完成！")
        return True
        
    except Exception as e:
        print(f"升級資料庫時發生錯誤: {str(e)}")
        return False
    finally:
        connection.close()

if __name__ == "__main__":
    upgrade_database() 
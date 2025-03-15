"""
資料庫清理工具 - 用於刪除過舊的職缺資料
"""

import argparse
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import DATABASE_URL

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("clean.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("db_cleaner")

def clean_old_jobs(older_than_days=180, dry_run=False):
    """
    清理過舊的職缺資料
    
    Args:
        older_than_days: 清理超過多少天的資料
        dry_run: 如果為 True，只顯示會刪除的記錄數量，不實際刪除
    
    Returns:
        deleted_count: 刪除的記錄數量
    """
    logger.info(f"開始清理超過 {older_than_days} 天的職缺資料...")
    
    # 計算截止日期
    cutoff_date = datetime.now() - timedelta(days=older_than_days)
    formatted_date = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')
    
    # 連接資料庫
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # 查詢將被刪除的記錄數量
        count_query = text(f"SELECT COUNT(*) FROM linkedin_jobs WHERE scrape_date < '{formatted_date}'")
        count_result = session.execute(count_query).scalar()
        
        logger.info(f"找到 {count_result} 筆超過 {older_than_days} 天的職缺資料")
        
        if dry_run:
            logger.info("乾跑模式：不實際刪除資料")
            return count_result
        
        if count_result > 0:
            # 執行刪除
            delete_query = text(f"DELETE FROM linkedin_jobs WHERE scrape_date < '{formatted_date}'")
            session.execute(delete_query)
            session.commit()
            logger.info(f"已刪除 {count_result} 筆過舊的職缺資料")
        else:
            logger.info("沒有過舊的職缺資料需要清理")
        
        return count_result
    except Exception as e:
        session.rollback()
        logger.error(f"清理資料時發生錯誤: {str(e)}")
        return 0
    finally:
        session.close()

def clean_by_keyword(keyword, dry_run=False):
    """
    根據關鍵字清理職缺資料
    
    Args:
        keyword: 要清理的關鍵字
        dry_run: 如果為 True，只顯示會刪除的記錄數量，不實際刪除
    
    Returns:
        deleted_count: 刪除的記錄數量
    """
    logger.info(f"開始清理關鍵字為 '{keyword}' 的職缺資料...")
    
    # 連接資料庫
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # 查詢將被刪除的記錄數量
        count_query = text(f"SELECT COUNT(*) FROM linkedin_jobs WHERE job_scrape LIKE '%{keyword}%'")
        count_result = session.execute(count_query).scalar()
        
        logger.info(f"找到 {count_result} 筆關鍵字為 '{keyword}' 的職缺資料")
        
        if dry_run:
            logger.info("乾跑模式：不實際刪除資料")
            return count_result
        
        if count_result > 0:
            # 執行刪除
            delete_query = text(f"DELETE FROM linkedin_jobs WHERE job_scrape LIKE '%{keyword}%'")
            session.execute(delete_query)
            session.commit()
            logger.info(f"已刪除 {count_result} 筆關鍵字為 '{keyword}' 的職缺資料")
        else:
            logger.info(f"沒有關鍵字為 '{keyword}' 的職缺資料需要清理")
        
        return count_result
    except Exception as e:
        session.rollback()
        logger.error(f"清理資料時發生錯誤: {str(e)}")
        return 0
    finally:
        session.close()

def clean_by_location(location, dry_run=False):
    """
    根據地點清理職缺資料
    
    Args:
        location: 要清理的地點
        dry_run: 如果為 True，只顯示會刪除的記錄數量，不實際刪除
    
    Returns:
        deleted_count: 刪除的記錄數量
    """
    logger.info(f"開始清理地點為 '{location}' 的職缺資料...")
    
    # 連接資料庫
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # 查詢將被刪除的記錄數量
        count_query = text(f"SELECT COUNT(*) FROM linkedin_jobs WHERE location LIKE '%{location}%'")
        count_result = session.execute(count_query).scalar()
        
        logger.info(f"找到 {count_result} 筆地點為 '{location}' 的職缺資料")
        
        if dry_run:
            logger.info("乾跑模式：不實際刪除資料")
            return count_result
        
        if count_result > 0:
            # 執行刪除
            delete_query = text(f"DELETE FROM linkedin_jobs WHERE location LIKE '%{location}%'")
            session.execute(delete_query)
            session.commit()
            logger.info(f"已刪除 {count_result} 筆地點為 '{location}' 的職缺資料")
        else:
            logger.info(f"沒有地點為 '{location}' 的職缺資料需要清理")
        
        return count_result
    except Exception as e:
        session.rollback()
        logger.error(f"清理資料時發生錯誤: {str(e)}")
        return 0
    finally:
        session.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LinkedIn 爬蟲資料庫清理工具')
    
    parser.add_argument('--older-than', type=int, default=180, help='清理超過多少天的資料 (預設: 180 天)')
    parser.add_argument('--keyword', type=str, help='清理指定關鍵字的資料')
    parser.add_argument('--location', type=str, help='清理指定地點的資料')
    parser.add_argument('--dry-run', action='store_true', help='乾跑模式：只顯示會刪除的記錄數量，不實際刪除')
    
    args = parser.parse_args()
    
    if args.keyword:
        clean_by_keyword(args.keyword, args.dry_run)
    elif args.location:
        clean_by_location(args.location, args.dry_run)
    else:
        clean_old_jobs(args.older_than, args.dry_run) 
"""
重置工具 - 用於重新初始化資料庫和爬蟲配置
"""

import argparse
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.database.models import init_db, Base
from app.database.operations import create_search_config, get_all_search_configs
from app.config import DATABASE_URL
from app.setup import setup_default_configs

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reset.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("reset_tool")

def reset_database(keep_configs=False):
    """
    重置資料庫 - 刪除並重新創建所有表
    
    Args:
        keep_configs: 如果為 True，會將現有的搜索配置備份並在重置後還原
    """
    logger.info("開始重置資料庫...")
    
    # 備份現有搜索配置（如果需要）
    config_backup = []
    if keep_configs:
        try:
            logger.info("備份現有搜索配置...")
            config_backup = get_all_search_configs(active_only=False)
            logger.info(f"已備份 {len(config_backup)} 個搜索配置")
        except Exception as e:
            logger.error(f"備份搜索配置時發生錯誤: {str(e)}")
            config_backup = []
    
    # 連接資料庫
    engine = create_engine(DATABASE_URL)
    
    # 刪除所有表
    try:
        logger.info("刪除所有資料表...")
        Base.metadata.drop_all(engine)
        logger.info("所有資料表已刪除")
    except Exception as e:
        logger.error(f"刪除資料表時發生錯誤: {str(e)}")
        return False
    
    # 重新創建所有表
    try:
        logger.info("重新創建所有資料表...")
        Base.metadata.create_all(engine)
        logger.info("所有資料表已重新創建")
    except Exception as e:
        logger.error(f"重新創建資料表時發生錯誤: {str(e)}")
        return False
    
    # 還原搜索配置（如果有備份）
    if keep_configs and config_backup:
        try:
            logger.info("還原搜索配置...")
            for config in config_backup:
                create_search_config(
                    name=config.name,
                    keyword=config.keyword,
                    location=config.location,
                    time_filter=config.time_filter,
                    max_pages=config.max_pages,
                    is_active=config.is_active
                )
            logger.info(f"已還原 {len(config_backup)} 個搜索配置")
        except Exception as e:
            logger.error(f"還原搜索配置時發生錯誤: {str(e)}")
    
    logger.info("資料庫重置完成")
    return True

def reset_and_setup():
    """重置資料庫並設置預設配置"""
    if reset_database(keep_configs=False):
        logger.info("設置預設搜索配置...")
        setup_default_configs(update_existing=False)
        logger.info("重置和設置完成")
        return True
    return False

def clear_jobs():
    """清空職缺資料但保留搜索配置"""
    logger.info("開始清空職缺資料...")
    
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # 清空 linkedin_jobs 表
        session.execute(text("DELETE FROM linkedin_jobs"))
        session.commit()
        logger.info("職缺資料已清空")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"清空職缺資料時發生錯誤: {str(e)}")
        return False
    finally:
        session.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LinkedIn 爬蟲重置工具')
    
    parser.add_argument('--reset-db', action='store_true', help='重置資料庫 (保留配置)')
    parser.add_argument('--reset-all', action='store_true', help='重置資料庫和配置')
    parser.add_argument('--clear-jobs', action='store_true', help='只清空職缺資料')
    parser.add_argument('--setup', action='store_true', help='設置預設配置')
    
    args = parser.parse_args()
    
    if args.reset_all:
        reset_database(keep_configs=False)
        if args.setup:
            setup_default_configs(update_existing=False)
    elif args.reset_db:
        reset_database(keep_configs=True)
    elif args.clear_jobs:
        clear_jobs()
    elif args.setup:
        setup_default_configs(update_existing=True)
    else:
        parser.print_help() 
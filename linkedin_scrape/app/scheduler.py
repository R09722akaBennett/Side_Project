import time
import schedule
import threading
from datetime import datetime
import logging

from app.scraper.linkedin import LinkedInScraper
from app.database.operations import get_all_search_configs
from app.config import SCHEDULE_INTERVAL

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("linkedin_scraper")

def run_all_search_configs():
    """執行所有活躍的搜索配置"""
    logger.info(f"開始執行所有搜索配置 - {datetime.now()}")
    
    # 獲取所有活躍的搜索配置
    configs = get_all_search_configs(active_only=True)
    
    if not configs:
        logger.warning("沒有找到活躍的搜索配置，請先添加配置")
        return
        
    logger.info(f"找到 {len(configs)} 個活躍配置，開始依序執行")
    
    scraper = LinkedInScraper(headless=True)
    
    for config in configs:
        try:
            logger.info(f"執行配置 '{config.name}': {config.keyword} 在 {config.location}")
            
            df = scraper.scrape_jobs(
                keyword=config.keyword,
                location=config.location,
                time_filter=config.time_filter,
                max_pages=config.max_pages,
                save_to_db=True
            )
            
            logger.info(f"配置 '{config.name}' 完成，共取得 {len(df)} 筆職缺")
            
            # 等待一段時間再執行下一個配置，避免 IP 被封
            time.sleep(30)
            
        except Exception as e:
            logger.error(f"執行配置 '{config.name}' 時發生錯誤: {str(e)}")

def run_scheduler():
    """運行排程器"""
    logger.info("啟動排程器，設置為每天中午12點執行所有搜索配置")
    
    # 設置定期執行排程 - 每天中午12點執行
    schedule.every().day.at("12:00").do(run_all_search_configs)
    
    # 檢查是否應該立即執行一次（首次啟動時）
    current_hour = datetime.now().hour
    current_minute = datetime.now().minute
    
    # 如果當前時間是中午12點之前，今天尚未執行
    # 如果當前時間是中午12點之後，等待明天執行
    if current_hour < 12 or (current_hour == 12 and current_minute == 0):
        logger.info("今天尚未到執行時間，將在今天中午12點執行")
    else:
        logger.info("今天的執行時間已過，將在明天中午12點執行")
    
    # 持續運行排程器
    while True:
        schedule.run_pending()
        time.sleep(60)  # 每分鐘檢查一次是否有任務需要執行

def start_scheduler_thread():
    """在背景執行緒中啟動排程器"""
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    return scheduler_thread 
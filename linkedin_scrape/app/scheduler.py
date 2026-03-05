import time
import schedule
import threading
from datetime import datetime, timedelta
import logging
import pytz

from app.scraper.linkedin import LinkedInScraper
from app.database.operations import get_all_search_configs, deduplicate_jobs
from app.processor import run_job_analysis
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

# 設置台灣時區
taiwan_tz = pytz.timezone('Asia/Taipei')

def run_all_search_configs():
    """執行所有活躍的搜索配置"""
    now = datetime.now(taiwan_tz)
    logger.info(f"開始執行所有搜索配置 - {now}")
    
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
    
    # 執行職缺去重，保留最新資料
    logger.info("所有配置爬取完成，開始執行資料去重...")
    try:
        deleted_count, kept_count = deduplicate_jobs()
        logger.info(f"資料去重完成: 已刪除 {deleted_count} 條重複記錄，保留 {kept_count} 條唯一記錄")
    except Exception as e:
        logger.error(f"執行資料去重時發生錯誤: {str(e)}")
        
    # 排程 1 小時後執行 AI 職缺分析
    logger.info("排程 1 小時後執行 AI 職缺分析...")
    
    # 計算 1 小時後的時間
    analysis_time = datetime.now() + timedelta(hours=1)
    analysis_time_str = analysis_time.strftime("%H:%M")
    logger.info(f"職缺分析將在 {analysis_time_str} 執行")
    
    # 創建唯一標識符的一次性任務
    job_id = f"job_analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # 使用 schedule 進行任務排程
    def job_wrapper():
        run_delayed_job_analysis()
        # 執行後取消該任務，確保只執行一次
        schedule.clear(job_id)
    
    # 創建定時任務並添加標籤
    schedule.every().day.at(analysis_time_str).do(job_wrapper).tag(job_id)

def run_delayed_job_analysis():
    """執行 AI 職缺分析"""
    logger.info("開始執行 AI 職缺分析...")
    
    try:
        # 執行職缺分析，分析最近 2 小時爬取的資料（確保覆蓋剛才爬取的資料）
        success = run_job_analysis(hours_ago=2)
        
        if success:
            logger.info("AI 職缺分析完成")
        else:
            logger.error("AI 職缺分析失敗")
            
    except Exception as e:
        logger.error(f"執行 AI 職缺分析時發生錯誤: {str(e)}")

def run_daily_deduplication():
    """每日執行職缺去重處理"""
    now = datetime.now(taiwan_tz)
    logger.info(f"開始執行每日資料去重處理 - {now}")
    
    try:
        deleted_count, kept_count = deduplicate_jobs()
        logger.info(f"每日資料去重完成: 已刪除 {deleted_count} 條重複記錄，保留 {kept_count} 條唯一記錄")
    except Exception as e:
        logger.error(f"執行每日資料去重時發生錯誤: {str(e)}")

def run_scheduler():
    """運行排程器"""
    logger.info("啟動排程器，設置為每72小時(3天)中午12點執行爬蟲，每日中午12點執行資料去重")
    
    # 設置爬蟲排程 - 每72小時執行一次，且在中午12點執行
    # 計算下一個執行時間點
    now = datetime.now(taiwan_tz)
    next_run_date = now
    
    # 如果當前時間已經過了12點，則設定為明天12點
    if now.hour >= 12:
        next_run_date = now + timedelta(days=1)
    
    # 設定時間為12:00
    next_run_date = next_run_date.replace(hour=12, minute=0, second=0, microsecond=0)
    
    # 計算初始延遲（從現在到下一個執行時間點的秒數）
    initial_delay_seconds = (next_run_date - now).total_seconds()
    
    # 第一次執行資訊
    logger.info(f"爬蟲首次執行時間: {next_run_date}, 延遲: {initial_delay_seconds/3600:.2f} 小時")
    
    # 設置爬蟲任務
    schedule.every(72).hours.at("12:00").do(run_all_search_configs)
    
    # 設置每日去重任務 - 每天中午12點執行
    schedule.every().day.at("12:00").do(run_daily_deduplication)
    
    # 如果當前時間與下一個執行時間的時間差小於等於1分鐘，立即執行一次
    if initial_delay_seconds <= 60:
        logger.info("當前時間接近排程時間，立即執行一次爬蟲")
        run_all_search_configs()
    
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
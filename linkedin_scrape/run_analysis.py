"""
執行職缺AI分析
"""
import os
import sys
import logging
from datetime import datetime

# 使用SQLite資料庫
os.environ["DATABASE_URL"] = "sqlite:///linkedin_jobs.db"

# 配置API金鑰(無代理)
os.environ["OPENAI_API_KEY"] = os.environ.get("X_AI_API_KEY", "sk-dummy-key")

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("job_analysis_runner")

if __name__ == "__main__":
    logger.info("開始執行職缺分析...")
    
    try:
        # 導入處理器(在環境變數設定後)
        from app.processor.job_analyzer import run_job_analysis
        
        # 默認分析過去24小時的資料
        hours_ago = 24
        
        # 如果提供命令行參數，則使用指定的小時數
        if len(sys.argv) > 1:
            try:
                hours_ago = int(sys.argv[1])
                logger.info(f"將分析過去 {hours_ago} 小時的職缺資料")
            except ValueError:
                logger.warning(f"無效的小時參數: {sys.argv[1]}，使用默認值24小時")
        
        # 執行分析
        start_time = datetime.now()
        logger.info(f"開始分析過去 {hours_ago} 小時爬取的職缺...")
        
        success = run_job_analysis(hours_ago=hours_ago)
        
        # 記錄完成時間
        end_time = datetime.now()
        duration = end_time - start_time
        
        if success:
            logger.info(f"職缺分析完成，耗時: {duration}")
        else:
            logger.error(f"職缺分析失敗，耗時: {duration}")
            
    except Exception as e:
        logger.error(f"執行職缺分析時發生錯誤: {str(e)}")
        sys.exit(1)

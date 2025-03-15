import logging
import argparse
from app.database.models import init_db, SearchConfig
from app.scheduler import start_scheduler_thread, run_all_search_configs
from app.scraper.linkedin import LinkedInScraper
from app.database.operations import (
    create_search_config, 
    get_all_search_configs, 
    update_search_config, 
    delete_search_config
)
from app.config import DEFAULT_KEYWORD, DEFAULT_LOCATION, DEFAULT_TIME_FILTER, DEFAULT_MAX_PAGES

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("linkedin_app")

def display_configs(configs):
    """顯示所有搜索配置"""
    if not configs:
        print("沒有找到任何搜索配置")
        return
        
    print(f"\n{'ID':^5}|{'名稱':^20}|{'關鍵字':^30}|{'地區':^20}|{'時間過濾':^12}|{'頁數':^6}|{'狀態':^6}")
    print("-" * 105)
    
    for config in configs:
        status = "啟用" if config.is_active else "禁用"
        print(f"{config.id:^5}|{config.name:^20}|{config.keyword:^30}|{config.location:^20}|{config.time_filter:^12}|{config.max_pages:^6}|{status:^6}")
    
    print("\n")

def main():
    """主程式入口點"""
    parser = argparse.ArgumentParser(description='LinkedIn 職缺爬蟲與自動排程')
    
    # 原有的參數
    parser.add_argument('--init', action='store_true', help='初始化資料庫')
    parser.add_argument('--scrape', action='store_true', help='立即執行爬蟲')
    parser.add_argument('--schedule', action='store_true', help='啟動排程器')
    parser.add_argument('--keyword', type=str, default=DEFAULT_KEYWORD, help='搜尋關鍵字')
    parser.add_argument('--location', type=str, default=DEFAULT_LOCATION, help='地點')
    parser.add_argument('--time-filter', type=str, default=DEFAULT_TIME_FILTER, help='時間過濾器')
    parser.add_argument('--max-pages', type=int, default=DEFAULT_MAX_PAGES, help='最大爬取頁數')
    
    # 新增搜索配置管理參數
    subparsers = parser.add_subparsers(dest='command', help='配置管理命令')
    
    # 列出所有搜索配置
    list_parser = subparsers.add_parser('list-configs', help='列出所有搜索配置')
    
    # 添加新的搜索配置
    add_parser = subparsers.add_parser('add-config', help='添加新的搜索配置')
    add_parser.add_argument('--name', type=str, required=True, help='配置名稱')
    add_parser.add_argument('--keyword', type=str, required=True, help='搜尋關鍵字')
    add_parser.add_argument('--location', type=str, required=True, help='地點')
    add_parser.add_argument('--time-filter', type=str, default=DEFAULT_TIME_FILTER, help='時間過濾器')
    add_parser.add_argument('--max-pages', type=int, default=DEFAULT_MAX_PAGES, help='最大爬取頁數')
    
    # 更新搜索配置
    update_parser = subparsers.add_parser('update-config', help='更新搜索配置')
    update_parser.add_argument('--id', type=int, required=True, help='配置 ID')
    update_parser.add_argument('--name', type=str, help='配置名稱')
    update_parser.add_argument('--keyword', type=str, help='搜尋關鍵字')
    update_parser.add_argument('--location', type=str, help='地點')
    update_parser.add_argument('--time-filter', type=str, help='時間過濾器')
    update_parser.add_argument('--max-pages', type=int, help='最大爬取頁數')
    update_parser.add_argument('--active', type=bool, help='是否啟用')
    
    # 刪除搜索配置
    delete_parser = subparsers.add_parser('delete-config', help='刪除搜索配置')
    delete_parser.add_argument('--id', type=int, required=True, help='配置 ID')
    
    # 立即執行所有配置
    run_all_parser = subparsers.add_parser('run-all-configs', help='立即執行所有搜索配置')
    
    args = parser.parse_args()
    
    # 初始化資料庫
    if args.init:
        logger.info("初始化資料庫...")
        engine = init_db()
        logger.info(f"資料庫初始化完成: {engine.url}")
    
    # 處理搜索配置命令
    if args.command == 'list-configs':
        configs = get_all_search_configs(active_only=False)
        display_configs(configs)
        
    elif args.command == 'add-config':
        config_id = create_search_config(
            name=args.name,
            keyword=args.keyword,
            location=args.location,
            time_filter=args.time_filter,
            max_pages=args.max_pages
        )
        if config_id:
            print(f"成功添加搜索配置，ID: {config_id}")
        else:
            print("添加搜索配置失敗")
            
    elif args.command == 'update-config':
        update_data = {}
        if args.name:
            update_data['name'] = args.name
        if args.keyword:
            update_data['keyword'] = args.keyword
        if args.location:
            update_data['location'] = args.location
        if args.time_filter:
            update_data['time_filter'] = args.time_filter
        if args.max_pages:
            update_data['max_pages'] = args.max_pages
        if args.active is not None:
            update_data['is_active'] = args.active
            
        success = update_search_config(args.id, **update_data)
        if success:
            print(f"成功更新搜索配置，ID: {args.id}")
        else:
            print(f"更新搜索配置失敗，ID: {args.id}")
            
    elif args.command == 'delete-config':
        success = delete_search_config(args.id)
        if success:
            print(f"成功刪除搜索配置，ID: {args.id}")
        else:
            print(f"刪除搜索配置失敗，ID: {args.id}")
            
    elif args.command == 'run-all-configs':
        print("開始執行所有搜索配置...")
        run_all_search_configs()
        print("所有配置執行完畢")
    
    # 立即執行爬蟲 (單一配置)
    elif args.scrape:
        logger.info("開始執行爬蟲...")
        scraper = LinkedInScraper(headless=True)
        df = scraper.scrape_jobs(
            keyword=args.keyword,
            location=args.location,
            time_filter=args.time_filter,
            max_pages=args.max_pages,
            save_to_db=True
        )
        logger.info(f"爬蟲完成，共取得 {len(df)} 筆職缺資料")
    
    # 啟動排程器 (自動執行所有活躍配置)
    elif args.schedule:
        logger.info("啟動排程器...")
        scheduler_thread = start_scheduler_thread()
        try:
            # 保持主程式運行
            scheduler_thread.join()
        except KeyboardInterrupt:
            logger.info("收到終止信號，停止應用程式...")
    
    # 如果沒有任何參數，顯示說明
    if not (args.init or args.scrape or args.schedule or args.command):
        parser.print_help()

if __name__ == "__main__":
    # 初始化資料庫
    init_db()
    
    # 執行主程式
    main() 
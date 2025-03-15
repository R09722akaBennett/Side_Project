from app.database.operations import create_search_config, get_all_search_configs, update_search_config, delete_search_config

def setup_default_configs(update_existing=True):
    """
    設置預設的搜索配置
    
    Args:
        update_existing: 如果為 True，將更新同名的已存在配置；如果為 False，只添加新配置
    """
    default_configs = [
        {
            "name": "AI Engineer in Taiwan",
            "keyword": "AI Engineer",
            "location": "Taiwan",
            "time_filter": "r86400",  # 一週內
            "max_pages": 5
        },
        {
            "name": "Data Scientist in Taiwan",
            "keyword": "Data Scientist",
            "location": "Taiwan",
            "time_filter": "r86400",
            "max_pages": 5
        },
        {
            "name": "AI Engineer in Singapore",
            "keyword": "AI Engineer",
            "location": "Singapore",
            "time_filter": "r86400",
            "max_pages": 5
        },
        {
            "name": "Data Scientist in Singapore",
            "keyword": "Data Scientist",
            "location": "Singapore",
            "time_filter": "r86400",
            "max_pages": 5
        },
        {
            "name": "Machine Learning Engineer in Taiwan",
            "keyword": "Machine Learning Engineer",
            "location": "Taiwan",
            "time_filter": "r86400",  # 24小時內
            "max_pages": 5
        },
        {
            "name": "Machine Learning Engineer in Singapore",
            "keyword": "Machine Learning Engineer",
            "location": "Singapore",
            "time_filter": "r86400",
            "max_pages": 5
        }
    ]
    
    # 獲取所有現有配置
    existing_configs = {config.name: config for config in get_all_search_configs(active_only=False)}
    
    created_configs = 0
    updated_configs = 0
    
    for config in default_configs:
        config_name = config['name']
        
        # 檢查是否已經存在同名配置
        if config_name in existing_configs and update_existing:
            # 更新現有配置
            config_id = existing_configs[config_name].id
            update_data = {
                'keyword': config['keyword'],
                'location': config['location'],
                'time_filter': config['time_filter'],
                'max_pages': config['max_pages'],
                'is_active': True
            }
            success = update_search_config(config_id, **update_data)
            if success:
                updated_configs += 1
                print(f"已更新配置: {config_name} (ID: {config_id})")
            else:
                print(f"更新配置失敗: {config_name}")
        elif config_name not in existing_configs:
            # 創建新配置
            config_id = create_search_config(**config)
            if config_id:
                created_configs += 1
                print(f"已創建配置: {config_name} (ID: {config_id})")
            else:
                print(f"創建配置失敗: {config_name}")
    
    print(f"成功創建 {created_configs} 個新配置，更新 {updated_configs} 個現有配置")

def run_specific_configs(config_names):
    """
    運行指定名稱的配置
    
    Args:
        config_names: 配置名稱列表
    """
    from app.scraper.linkedin import LinkedInScraper
    
    # 獲取所有配置
    all_configs = get_all_search_configs(active_only=False)
    
    # 篩選出指定名稱的配置
    configs_to_run = [c for c in all_configs if c.name in config_names]
    
    if not configs_to_run:
        print(f"未找到指定的配置: {config_names}")
        return
    
    print(f"將執行以下 {len(configs_to_run)} 個配置:")
    for config in configs_to_run:
        print(f"- {config.name} (關鍵字: {config.keyword}, 地點: {config.location})")
    
    # 執行所選配置
    scraper = LinkedInScraper(headless=True)
    for config in configs_to_run:
        print(f"\n開始執行配置: {config.name}")
        df = scraper.scrape_jobs(
            keyword=config.keyword,
            location=config.location,
            time_filter=config.time_filter,
            max_pages=config.max_pages,
            save_to_db=True
        )
        print(f"配置 {config.name} 執行完成，取得 {len(df)} 個職缺")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LinkedIn 職缺爬蟲設置工具')
    parser.add_argument('--setup', action='store_true', help='設置預設配置')
    parser.add_argument('--update', action='store_true', help='更新現有配置')
    parser.add_argument('--run', nargs='+', help='執行指定名稱的配置，多個配置名稱用空格分隔')
    parser.add_argument('--list', action='store_true', help='列出所有配置')
    
    args = parser.parse_args()
    
    if args.setup:
        # 設置預設配置（不更新現有配置）
        setup_default_configs(update_existing=False)
    elif args.update:
        # 設置預設配置（更新現有配置）
        setup_default_configs(update_existing=True)
    elif args.run:
        # 執行指定的配置
        run_specific_configs(args.run)
    elif args.list:
        # 列出所有配置
        configs = get_all_search_configs(active_only=False)
        print(f"\n{'ID':^5}|{'名稱':^30}|{'關鍵字':^30}|{'地區':^20}|{'時間過濾':^12}|{'頁數':^6}|{'狀態':^6}")
        print("-" * 115)
        for config in configs:
            status = "啟用" if config.is_active else "禁用"
            print(f"{config.id:^5}|{config.name:^30}|{config.keyword:^30}|{config.location:^20}|{config.time_filter:^12}|{config.max_pages:^6}|{status:^6}")
    else:
        # 如果沒有參數，顯示幫助
        parser.print_help() 
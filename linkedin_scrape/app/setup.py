from app.database.operations import create_search_config

def setup_default_configs():
    """設置預設的搜索配置"""
    default_configs = [
        {
            "name": "AI Engineer in Taiwan",
            "keyword": "AI Engineer",
            "location": "Taiwan",
            "time_filter": "r604800",  # 一週內
            "max_pages": 5
        },
        {
            "name": "Data Scientist in Taiwan",
            "keyword": "Data Scientist",
            "location": "Taiwan",
            "time_filter": "r604800",
            "max_pages": 5
        },
        # {
        #     "name": "Software Engineer in Taipei",
        #     "keyword": "Software Engineer",
        #     "location": "Taipei",
        #     "time_filter": "r604800",
        #     "max_pages": 5
        # },
        {
            "name": "Machine Learning in Taiwan",
            "keyword": "Machine Learning Engineer",
            "location": "Taiwan",
            "time_filter": "r604800",
            "max_pages": 5
        }
    ]
    
    created_configs = 0
    for config in default_configs:
        config_id = create_search_config(**config)
        if config_id:
            created_configs += 1
            print(f"已創建配置: {config['name']} (ID: {config_id})")
    
    print(f"成功創建 {created_configs}/{len(default_configs)} 個預設配置")

if __name__ == "__main__":
    setup_default_configs() 
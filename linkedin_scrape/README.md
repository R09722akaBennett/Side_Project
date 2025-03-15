# LinkedIn Job Scraper

A modular Python application for scraping job listings from LinkedIn.

## Features

- Search for jobs by keyword and location
- Filter jobs by time period
- Extract detailed job information including:
  - Company name
  - Job title
  - Job description
  - Seniority level
  - Employment type
  - Job function
  - Industries
  - Posting date
  - Job Search Keyword
  - Location Search Parameter

## Directory Structure

```
linkedin_scraper/
├── config/             # Configuration settings
├── scrapers/           # Job scraping modules
├── utils/              # Utility functions
├── __init__.py         # Package initialization
├── main.py             # Main entry point
output/                 # Output directory for CSV files
requirements.txt        # Project dependencies
README.md               # This file
```

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script with:

```python
python linkedin_scraper/main.py
```

Or import and use the scraper in your own code:

```python
from linkedin_scraper.scrapers.linkedin import LinkedInJobScraper

scraper = LinkedInJobScraper()
jobs_df = scraper.scrape(
    input_keyword='AI Engineer',
    location='Taiwan',
    time_filter='r86400',
    max_pages=5
)
jobs_df.to_csv('output/linkedin_jobs.csv', index=False)
```

## Configuration

Edit `linkedin_scraper/config/settings.py` to modify default parameters. 

## Service Initialization

To initialize the scraping service, ensure that your environment variables are set correctly in the `.env` file. This includes setting up your LinkedIn credentials and any other necessary API keys.

## Database Setup

1. Ensure your database is running and accessible.
2. Initialize the database by running the following command:
   ```bash
   python linkedin_scraper/init_db.py
   ```

## Pulling Data from the Database

To retrieve data from the database, you can use the following script:

```python
from linkedin_scraper.database import Database

db = Database()
jobs_data = db.get_all_jobs()
print(jobs_data)
```

This will fetch all job listings stored in the database and print them to the console.

## Docker Deployment

To deploy the application using Docker, use the provided `docker-compose.yml` file. Run the following command to start the services:

```bash
docker-compose up --build
```

Ensure that your Docker environment is set up correctly and that all necessary environment variables are configured in the `.env` file.

## 資料存儲與維護

### 增量爬取與資料儲存

本系統設計確保每次爬蟲執行時：
1. 所有新爬取的職缺會**自動儲存**到資料庫中
2. 系統使用職缺 URL 作為**唯一識別符**，確保不會儲存重複的職缺
3. 每次爬蟲執行時**只會新增**資料庫中尚未存在的職缺，不會覆蓋既有資料

這意味著您可以安全地多次執行爬蟲，包括使用排程器每天自動執行，系統會不斷累積新的職缺資訊，而不會丟失之前的資料。

### 資料庫重置與初始化

如果您需要重新初始化資料庫或清空特定資料，可以使用 `app/reset.py` 工具：

```bash
# 列出所有可用選項
docker exec -it linkedin-scraper python -m app.reset --help

# 清空職缺資料但保留搜索配置
docker exec -it linkedin-scraper python -m app.reset --clear-jobs

# 重置資料庫但保留搜索配置
docker exec -it linkedin-scraper python -m app.reset --reset-db

# 完全重置資料庫（包括所有配置和職缺資料）
docker exec -it linkedin-scraper python -m app.reset --reset-all

# 重置後，重新設置預設配置
docker exec -it linkedin-scraper python -m app.reset --reset-all --setup
```

### 定期維護建議

為了維持系統的效能和資料庫的健康，建議：

1. **定期備份**：每月備份資料庫，避免資料遺失
2. **定期清理**：每 3-6 個月清理過舊的職缺資料，保持資料庫輕量
3. **監控日誌**：定期檢查 `scraper.log` 和 `reset.log`，確保系統正常運行

您可以設置定期任務來執行這些維護工作：

```bash
# 例如，每月第一天清理超過 6 個月的職缺資料
0 0 1 * * docker exec -it linkedin-scraper python -m app.database.clean --older-than=180
```

## 使用 setup.py 管理爬取內容

`app/setup.py` 是一個方便的管理工具，用於設置、更新和執行職缺搜索配置。您可以在此文件中集中管理您要爬取的職缺內容，並且能夠輕鬆地同步更新到資料庫。

### 編輯爬取配置

打開 `app/setup.py` 文件，在 `default_configs` 列表中添加或修改配置：

```python
default_configs = [
    {
        "name": "AI Engineer in Taiwan",
        "keyword": "AI Engineer",
        "location": "Taiwan",
        "time_filter": "r604800",  # 一週內
        "max_pages": 5
    },
    # 在此添加更多配置...
]
```

每個配置包含以下參數：
- `name`: 配置的唯一名稱
- `keyword`: 要搜索的職缺關鍵字
- `location`: 地點
- `time_filter`: 時間過濾器 (r86400=24小時內, r604800=一週內)
- `max_pages`: 爬取的最大頁數

### 命令行工具

`setup.py` 支持多種命令行參數，讓您能夠方便地管理配置：

1. **列出所有配置**：
   ```bash
   docker exec -it linkedin-scraper python -m app.setup --list
   ```

2. **設置新配置** (不更新已存在的同名配置)：
   ```bash
   docker exec -it linkedin-scraper python -m app.setup --setup
   ```

3. **更新所有配置** (更新已存在的同名配置)：
   ```bash
   docker exec -it linkedin-scraper python -m app.setup --update
   ```

4. **執行特定配置** (按名稱執行一個或多個配置)：
   ```bash
   docker exec -it linkedin-scraper python -m app.setup --run "AI Engineer in Taiwan" "Data Scientist in Singapore"
   ```

### 使用流程

典型的使用流程如下：

1. 修改 `app/setup.py` 中的 `default_configs` 列表，添加您需要的職缺搜索條件
2. 運行 `--update` 命令同步更新配置到資料庫
3. 使用 `--list` 命令確認配置已正確更新
4. 使用 `--run` 命令執行特定配置，或使用 `main.py` 的 `run-all-configs` 命令執行所有啟用的配置

### 與一般爬蟲命令的區別

相比於直接使用 `main.py --scrape` 命令，`setup.py` 提供了以下優勢：
- 在一個文件中集中管理多個搜索配置
- 批量添加或更新配置
- 能夠使用名稱來選擇執行特定配置
- 便於維護和版本控制配置

## 調整爬取職缺的方法

您有多種方式可以調整爬取職缺的設定：

### 1. 透過命令行直接爬取

您可以使用以下命令直接執行爬蟲，並指定所需參數：

```bash
docker exec -it linkedin-scraper python -m app.main --scrape --keyword="您的關鍵字" --location="地點" --time-filter="時間過濾器" --max-pages=頁數
```

參數說明：
- `--keyword`：搜尋關鍵字，例如 "Data Analyst"、"Software Engineer"
- `--location`：搜尋地點，例如 "Taiwan"、"Taipei"、"Remote"
- `--time-filter`：時間過濾器，例如 `r86400`（24小時內）、`r604800`（一週內）
- `--max-pages`：最大爬取頁數，每頁約 25 個職缺

範例：
```bash
docker exec -it linkedin-scraper python -m app.main --scrape --keyword="Data Analyst" --location="Taiwan" --max-pages=3
```

### 2. 透過搜索配置管理

您可以創建、更新和管理搜索配置，這樣就可以一次設定好搜索條件，之後重複使用：

- **列出所有配置**：
  ```bash
  docker exec -it linkedin-scraper python -m app.main list-configs
  ```

- **添加新配置**：
  ```bash
  docker exec -it linkedin-scraper python -m app.main add-config --name "配置名稱" --keyword "關鍵字" --location "地點" --time-filter "時間過濾器" --max-pages 頁數
  ```

- **更新配置**：
  ```bash
  docker exec -it linkedin-scraper python -m app.main update-config --id 配置ID --keyword "新關鍵字" --location "新地點"
  ```

- **刪除配置**：
  ```bash
  docker exec -it linkedin-scraper python -m app.main delete-config --id 配置ID
  ```

- **執行所有配置**：
  ```bash
  docker exec -it linkedin-scraper python -m app.main run-all-configs
  ```

### 3. 啟動排程器自動執行

您可以啟動排程器，它會按照設定的時間自動執行所有啟用狀態的配置：

```bash
docker exec -it linkedin-scraper python -m app.main --schedule
```

## 執行流程說明

以下是整個系統的執行流程：

```
+-------------------+     +-------------------+     +-------------------+
| 初始化資料庫       | --> | 添加搜索配置       | --> | 執行爬蟲           |
| (--init)          |     | (add-config)      |     | (--scrape 或      |
+-------------------+     +-------------------+     |  run-all-configs) |
                                                   +-------------------+
                                                           |
                                                           v
+-------------------+     +-------------------+     +-------------------+
| 結果儲存到資料庫   | <-- | 爬取職缺詳細資訊   | <-- | 爬取搜索結果頁面   |
| (自動處理)        |     | (自動處理)        |     | (自動處理)        |
+-------------------+     +-------------------+     +-------------------+
        |
        v
+-------------------+     +-------------------+
| 查詢資料           | --> | 數據分析和報表     |
| (另外實作)        |     | (另外實作)        |
+-------------------+     +-------------------+
```

### 完整流程說明：

1. **初始化資料庫**：首次使用前，初始化資料庫結構
   ```bash
   docker exec -it linkedin-scraper python -m app.main --init
   ```

2. **添加搜索配置**：設定您想要搜索的職缺條件
   ```bash
   docker exec -it linkedin-scraper python -m app.main add-config --name "數據分析師" --keyword "Data Analyst" --location "Taiwan"
   ```

3. **執行爬蟲**：有兩種方式
   - 直接執行特定搜索：
     ```bash
     docker exec -it linkedin-scraper python -m app.main --scrape --keyword="Data Analyst" --location="Taiwan"
     ```
   - 執行所有已保存的配置：
     ```bash
     docker exec -it linkedin-scraper python -m app.main run-all-configs
     ```

4. **自動化排程**：設定自動化執行
   ```bash
   docker exec -it linkedin-scraper python -m app.main --schedule
   ```

5. **數據查詢**：您可以自行開發查詢界面或直接使用資料庫工具來查詢已收集的職缺資訊。

### 升級資料庫（如需添加新欄位）：

如果您需要升級資料庫結構，可以使用以下命令：
```bash
docker exec -it linkedin-scraper python -m app.main --upgrade
```

## Local Deployment with Scheduled Scraping

You can deploy this application locally and set it to run automatically at 12:00 PM (noon) every day.

### For Linux/Mac Users:

1. Make the startup script executable:
   ```bash
   chmod +x start_local_scheduler.sh
   chmod +x stop_scheduler.sh
   ```

2. Start the scheduler:
   ```bash
   ./start_local_scheduler.sh
   ```

3. To stop the scheduler:
   ```bash
   ./stop_scheduler.sh
   ```

### For Windows Users:

1. Open PowerShell as Administrator.

2. If you haven't already, you may need to set the execution policy to allow running scripts:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. Run the startup script:
   ```powershell
   .\start_local_scheduler.ps1
   ```

4. To stop the scheduler:
   ```powershell
   .\stop_scheduler.ps1
   ```

### Checking Logs

The scheduler logs are saved to `scheduler.log`. You can view them by running:
```bash
tail -f scheduler.log  # Linux/Mac
```
Or on Windows:
```powershell
Get-Content scheduler.log -Wait
```

### Managing Search Configurations

Before the scheduler can run effectively, you need to add at least one search configuration:

```bash
python -m app.main add-config --name "Data Science Jobs" --keyword "Data Scientist" --location "Taiwan" --max-pages 5
```

You can list all configurations:
```bash
python -m app.main list-configs
```

And remove configurations you no longer need:
```bash
python -m app.main delete-config --id <config_id>
``` 
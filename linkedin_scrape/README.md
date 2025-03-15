# LinkedIn Job Scraper

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.9%2B-green.svg)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)

A powerful LinkedIn job scraping system for automatically collecting, analyzing, and monitoring job opportunities on LinkedIn.

(一個強大的LinkedIn職缺爬蟲系統，用於自動收集、分析和監控LinkedIn上的工作機會。)

## 📌 Project Overview | 專案概述

The LinkedIn Job Scraper automatically retrieves job information from LinkedIn based on specified keywords and locations, storing them in a database for analysis. This system is particularly useful for job seekers, HR professionals, and market analysts monitoring employment market trends.

(LinkedIn職缺爬蟲系統能自動從LinkedIn獲取指定關鍵字和地點的職缺資訊，並存入資料庫以供分析。本系統特別適合求職者、人力資源專業人員和市場分析師監控就業市場趨勢。)

### 🌟 Core Features | 核心功能

- **Automated Job Scraping**: Scrape LinkedIn jobs by keyword, location, and time range
  (自動化職缺爬取：依關鍵字、地點和時間範圍爬取LinkedIn職缺)
- **Flexible Search Configurations**: Create and manage multiple search criteria
  (彈性搜尋配置：建立和管理多個搜尋條件，按需執行)
- **Scheduled Execution**: Set up timed schedules for automatic execution
  (排程自動執行：設定定時排程，自動執行爬蟲任務)
- **Data Persistence**: Store scraped data in PostgreSQL database, avoiding duplicates
  (資料持久化：將爬取資料存入PostgreSQL資料庫，避免重複)
- **Detailed Job Information**: Collect complete job information including title, company, description
  (詳細職缺資訊：收集職缺標題、公司、描述、資歷要求等完整資訊)
- **Docker Containerization**: Simplify deployment and environment management
  (Docker容器化：簡化部署和環境管理)
- **Health Monitoring**: System status monitoring and logging
  (健康監控：系統運行狀態監控和日誌記錄)

## 🏗 System Architecture | 系統架構

```mermaid
graph TD
    User[User/使用者] --> CLI[Command Line Interface/命令行介面]
    CLI --> Scheduler[Scheduler/排程器]
    CLI --> Scraper[Scraper Engine/爬蟲引擎]
    Scheduler --> Scraper
    Scraper --> Database[(PostgreSQL Database/資料庫)]
    Setup[Setup Tool/設定工具] --> Database
    Config[Search Configurations/搜尋配置] <--> Database
    Health[Health Monitoring/健康監控] --> Scheduler
    Health --> Scraper
    Health --> Database
```

## 📁 Project Structure | 專案結構

```
linkedin-scraper/
├── app/                  # Application main directory | 應用程式主目錄
│   ├── api/              # API services | API服務
│   ├── config/           # Configuration settings | 配置設定
│   ├── database/         # Database models and operations | 資料庫模型與操作
│   │   ├── models.py     # Data models | 資料模型
│   │   ├── operations.py # Database operations | 資料庫操作
│   │   └── upgrade_db.py # Database upgrade script | 資料庫升級腳本
│   ├── scraper/          # Scraper core components | 爬蟲核心組件
│   │   ├── linkedin.py   # LinkedIn scraper implementation | LinkedIn爬蟲實現
│   │   └── utils.py      # Scraper utility functions | 爬蟲工具函數
│   ├── monitoring/       # Monitoring components | 監控組件
│   ├── scheduler/        # Scheduler components | 排程組件
│   ├── utils/            # General utilities | 通用工具
│   ├── main.py           # Main program entry | 主程式入口
│   └── setup.py          # Setup script | 設置腳本
├── docker/               # Docker related files | Docker相關文件
│   ├── Dockerfile        # Container definition | 容器定義
│   └── docker-compose.yml # Container orchestration | 容器編排
├── scripts/              # Script files | 腳本文件
├── requirements.txt      # Python dependencies | Python依賴
├── migration.md          # GCP migration guide | GCP遷移指南
└── README.md             # This file | 本文件
```

## 🚀 Quick Start | 快速開始

### Prerequisites | 前置需求

- Docker and Docker Compose | Docker與Docker Compose
- Python 3.9+ (for local development | 本地開發)
- PostgreSQL (for local development, included in Docker | 本地開發，Docker中已包含)

### Running with Docker (Recommended) | 使用Docker運行（推薦）

1. **Clone the project | 複製專案**
   ```bash
   git clone https://github.com/yourusername/linkedin-scraper.git
   cd linkedin-scraper
   ```

2. **Start Docker container | 啟動Docker容器**
   ```bash
   docker-compose up -d
   ```

3. **Initialize database | 初始化資料庫**
   ```bash
   docker exec -it linkedin-scraper python -m app.main --init
   ```

4. **Upgrade database** (if adding new fields | 如需添加新欄位)
   ```bash
   docker exec -it linkedin-scraper python -m app.main --upgrade
   ```

### Local Development Environment Setup | 本地開發環境設置

1. **Install dependencies | 安裝依賴**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables | 設置環境變數**
   ```bash
   export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/linkedin
   ```

3. **Initialize database | 初始化資料庫**
   ```bash
   python -m app.main --init
   ```

## 💻 Usage Guide | 使用指南

### 1. Managing Search Configurations | 管理搜尋配置

The LinkedIn scraper supports two ways to manage configurations: direct command line and setup.py.

(LinkedIn爬蟲支持兩種管理配置的方式：直接命令行和setup.py。)

#### Using setup.py (Recommended) | 使用setup.py管理（推薦）

Edit the `default_configs` list in the `app/setup.py` file:

(編輯`app/setup.py`文件的`default_configs`列表：)

```python
default_configs = [
    {
        "name": "AI Engineer in Taiwan",
        "keyword": "AI Engineer",
        "location": "Taiwan",
        "time_filter": "r604800",  # Within one week | 一週內
        "max_pages": 5
    },
    # Add more configurations... | 添加更多配置...
]
```

Then run the following command to update configurations:

(然後運行以下命令更新配置：)

```bash
docker exec -it linkedin-scraper python -m app.setup --update
```

List all configurations:
(列出所有配置：)
```bash
docker exec -it linkedin-scraper python -m app.setup --list
```

Run specific configurations:
(運行特定配置：)
```bash
docker exec -it linkedin-scraper python -m app.setup --run "AI Engineer in Taiwan"
```

#### Using Command Line | 使用命令行管理

List all configurations:
(列出所有配置：)
```bash
docker exec -it linkedin-scraper python -m app.main list-configs
```

Add new configuration:
(添加新配置：)
```bash
docker exec -it linkedin-scraper python -m app.main add-config --name "Data Scientist" --keyword "Data Scientist" --location "Taiwan" --max-pages 3
```

Update configuration:
(更新配置：)
```bash
docker exec -it linkedin-scraper python -m app.main update-config --id 1 --keyword "ML Engineer" --location "Remote"
```

Delete configuration:
(刪除配置：)
```bash
docker exec -it linkedin-scraper python -m app.main delete-config --id 1
```

### 2. Running the Scraper | 執行爬蟲

#### Direct Execution for Specific Search | 直接執行特定搜尋

```bash
docker exec -it linkedin-scraper python -m app.main --scrape --keyword="Data Analyst" --location="Taiwan" --max-pages=3
```

#### Run All Enabled Configurations | 執行所有啟用的配置

```bash
docker exec -it linkedin-scraper python -m app.main run-all-configs
```

#### Start Scheduler for Automatic Execution | 啟動排程自動執行

```bash
docker exec -d linkedin-scraper python -m app.main --schedule
```

### 3. Parameter Explanation | 參數說明

- `--keyword`: Search keyword, e.g., "Data Analyst", "Software Engineer"
  (搜尋關鍵字，例如 "Data Analyst"、"Software Engineer")
- `--location`: Search location, e.g., "Taiwan", "Taipei", "Remote"
  (搜尋地點，例如 "Taiwan"、"Taipei"、"Remote")
- `--time-filter`: Time filter
  (時間過濾器)
  - `r86400`: Within 24 hours | 24小時內
  - `r604800`: Within one week | 一週內
  - `r2592000`: Within one month | 一個月內
- `--max-pages`: Maximum pages to scrape, each page contains about 25 jobs
  (最大爬取頁數，每頁約25個職缺)

## 📊 Execution Flow | 執行流程

```mermaid
sequenceDiagram
    participant U as User/使用者
    participant C as Command Line/命令行
    participant S as Scheduler/排程器
    participant SC as Scraper Engine/爬蟲引擎
    participant DB as Database/資料庫

    Note over U,DB: Initialization Phase/初始化階段
    U->>C: Initialize database/初始化資料庫 (--init)
    C->>DB: Create table structure/創建資料表結構
    DB-->>C: Initialization complete/初始化完成
    
    Note over U,DB: Configuration Phase/配置階段
    U->>C: Add search configuration/添加搜索配置
    C->>DB: Store configuration/存儲配置
    
    Note over U,DB: Execution Phase/執行階段
    U->>C: Run scraper/執行爬蟲 (--scrape or run-all-configs)
    C->>SC: Start scraping task/啟動爬蟲任務
    SC->>SC: Scrape LinkedIn search results/爬取LinkedIn搜索結果
    SC->>SC: Parse job details/解析職缺詳情
    SC->>DB: Store job data/存儲職缺數據
    
    Note over U,DB: Scheduling Phase/排程階段
    U->>C: Start scheduler/啟動排程器 (--schedule)
    C->>S: Initialize scheduler/初始化排程器
    S->>S: Wait for scheduled time/等待排程時間
    S->>SC: Trigger scraping task/觸發爬蟲任務
    SC->>DB: Store new jobs/存儲新職缺
    
    Note over U,DB: Query Phase/查詢階段
    U->>DB: Query collected job data/查詢收集的職缺數據
```

## 🔍 Monitoring System | 監控系統

### Checking Running Status | 檢查運行狀態

```bash
# Check container status | 查看容器狀態
docker ps | grep linkedin-scraper

# View logs | 查看日誌
docker logs -f linkedin-scraper

# Check recently scraped jobs count | 查看最近爬取的職缺數量
docker exec -it linkedin-scraper python -c "
from app.database.operations import get_db_session
from app.database.models import LinkedInJob
from datetime import datetime, timedelta
import sqlalchemy

session = get_db_session()
yesterday = datetime.now() - timedelta(days=1)
recent_jobs = session.query(LinkedInJob).filter(LinkedInJob.scrape_date >= yesterday).count()
print(f'Jobs scraped in the last 24 hours | 過去24小時內爬取的職缺數量: {recent_jobs}')
session.close()
"
```

## ☁️ Cloud Deployment | 雲端部署

This system can be deployed to Google Cloud Platform. See the [GCP Migration Guide](migration.md) for detailed steps.

(本系統可以部署到Google Cloud Platform，詳細步驟請查看[GCP遷移指南](migration.md)。)

## ⚠️ Notes | 注意事項

1. **Scraping Frequency**: Do not set too high a scraping frequency to avoid being restricted by LinkedIn
   (爬蟲執行頻率：請勿設置過高的爬取頻率，以避免被LinkedIn限制)
2. **Resource Usage**: The scraper will consume certain CPU and memory resources during execution
   (資源使用：爬蟲執行時會消耗一定CPU和記憶體資源)
3. **Network Connection**: Ensure the system has a stable network connection
   (網絡連接：確保系統有穩定的網絡連接)
4. **Data Retention**: The system automatically deduplicates data, only adding new jobs without deleting old data
   (資料保留：系統會自動去重，只添加新職缺，不會刪除舊資料)

## 🤝 Contribution Guide | 貢獻指南

Contributions of code, issue reports, or improvement suggestions are welcome. Please follow these steps:

(歡迎貢獻代碼、報告問題或提出改進建議。請遵循以下步驟：)

1. Fork the project | Fork專案
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License | 許可證

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

(本專案採用MIT許可證 - 查看[LICENSE](LICENSE)文件了解更多詳情。)
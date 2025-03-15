# LinkedIn爬蟲專案 GCP遷移指南

本文檔提供了將LinkedIn爬蟲專案從本地環境遷移到Google Cloud Platform (GCP)的詳細步驟。

## 目錄

1. [架構概述](#架構概述)
2. [GCP服務選擇](#gcp服務選擇)
3. [前置準備](#前置準備)
4. [設置雲端資料庫](#設置雲端資料庫)
5. [容器化應用程序](#容器化應用程序)
6. [部署到Cloud Run](#部署到cloud-run)
7. [設置定時執行](#設置定時執行)
8. [監控和日誌](#監控和日誌)
9. [成本與安全考量](#成本與安全考量)
10. [故障排除](#故障排除)

## 架構概述

雲端部署架構將由以下組件組成：

- **爬蟲應用**：容器化後部署到Cloud Run
- **資料庫**：使用Cloud SQL PostgreSQL存儲爬取的職缺數據
- **定時任務**：使用Cloud Scheduler觸發爬蟲執行
- **監控系統**：使用Cloud Monitoring追踪應用運行狀態
- **安全管理**：使用Secret Manager存儲敏感資訊

![GCP部署架構圖](https://mermaid.ink/img/pako:eNp1UsFu2zAM_RVCpxao7SRpUw9FD0aHdQOGdheCHmjJdITIkkZRQdx0_z5KTrYBxXqRqCfy8fGRO-E7RoILUcYqE8Iw0bZY1QaJLtpBbT9rQq4TwepRm_ZBkNk2-8rQ_WCvDQnBK3TCNpQ3mZxdyI-HYKdx-GRajW0wgX7Aun-PqDp8lRlWb7LPpJuD54XoAm1pJsRGNfCBXm6VxfZ-7xrFMwAvsXfjYEbXnN34Aie-nWGqMcMVHBMlR37JMDfKYQu7Fsl4frY_-CflV-qYLBbk4_kMYDKLCVHaNWP5nFKIYlMqSz6DYV14xYmjWuNhyOGN7jQ4WnIBcQd3K7y-lw-vjXgLHxpjNT3XmG-UIYdN5QytOeEVuhI7xeFZl6i1MhdcQ0IM2uXIPpbBcHGkMRDsekwDQPJgL1Y-crHXKRUr_ZPLPj_82Pw_qd3tZnpxHuPJWOSJZEeOk4mEkQTCTwxlbzQbpuYGkhqq3yW5tJP-Q96lRGCVLPFVE3i3JZfFIBTxJK-LXH7mseYx3vIsCEXBn9V7wcc8lfIp3HLB67xQz3_LvMWXTDanI8MX2L-Dsg0?type=png)

## GCP服務選擇

為LinkedIn爬蟲專案，我們將使用以下GCP服務：

- **Cloud Run**：無服務器容器平台，按使用量計費，適合間歇性執行的爬蟲
- **Cloud SQL**：托管的關聯型資料庫服務，提供PostgreSQL支持
- **Cloud Storage**：用於存儲備份和日誌文件
- **Cloud Scheduler**：定時觸發爬蟲作業
- **Cloud Monitoring**：監控應用性能和健康狀態
- **Secret Manager**：安全存儲敏感信息如資料庫密碼

## 前置準備

### 安裝並設置Google Cloud SDK

1. 下載並安裝 [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)

2. 登入您的Google帳戶：
   ```bash
   gcloud auth login
   ```

### 創建GCP專案

1. 創建新專案：
   ```bash
   gcloud projects create linkedin-scraper-project --name="LinkedIn Job Scraper"
   ```

2. 設置為當前專案：
   ```bash
   gcloud config set project linkedin-scraper-project
   ```

### 啟用所需GCP服務

```bash
gcloud services enable cloudbuild.googleapis.com run.googleapis.com sql.googleapis.com \
   cloudscheduler.googleapis.com monitoring.googleapis.com secretmanager.googleapis.com \
   storage.googleapis.com
```

### 創建服務帳號

1. 創建服務帳號：
   ```bash
   gcloud iam service-accounts create linkedin-scraper-sa \
      --description="LinkedIn Scraper Service Account" \
      --display-name="linkedin-scraper-sa"
   ```

2. 賦予必要權限：
   ```bash
   # Cloud SQL訪問權限
   gcloud projects add-iam-policy-binding linkedin-scraper-project \
      --member="serviceAccount:linkedin-scraper-sa@linkedin-scraper-project.iam.gserviceaccount.com" \
      --role="roles/cloudsql.client"
   
   # Secret Manager訪問權限
   gcloud projects add-iam-policy-binding linkedin-scraper-project \
      --member="serviceAccount:linkedin-scraper-sa@linkedin-scraper-project.iam.gserviceaccount.com" \
      --role="roles/secretmanager.secretAccessor"
   
   # Cloud Storage訪問權限
   gcloud projects add-iam-policy-binding linkedin-scraper-project \
      --member="serviceAccount:linkedin-scraper-sa@linkedin-scraper-project.iam.gserviceaccount.com" \
      --role="roles/storage.objectAdmin"
   ```

## 設置雲端資料庫

### 創建Cloud SQL實例

```bash
gcloud sql instances create linkedin-db \
   --database-version=POSTGRES_13 \
   --tier=db-f1-micro \
   --region=asia-east1 \
   --root-password="YOUR_SECURE_ROOT_PASSWORD"
```

### 創建資料庫和用戶

```bash
# 創建資料庫
gcloud sql databases create linkedin_jobs --instance=linkedin-db

# 創建用戶
gcloud sql users create linkedin_user \
   --instance=linkedin-db \
   --password="YOUR_USER_PASSWORD"
```

### 將密碼存儲到Secret Manager

```bash
echo -n "YOUR_USER_PASSWORD" | gcloud secrets create linkedin-db-password \
   --data-file=- \
   --replication-policy="automatic"
```

### 為服務帳號授予訪問Secret的權限

```bash
gcloud secrets add-iam-policy-binding linkedin-db-password \
   --member="serviceAccount:linkedin-scraper-sa@linkedin-scraper-project.iam.gserviceaccount.com" \
   --role="roles/secretmanager.secretAccessor"
```

## 容器化應用程序

### 修改應用配置

創建或修改 `app/config/environments.py` 文件：

```python
import os

# 環境設置
ENV = os.environ.get('ENVIRONMENT', 'development')

# 數據庫配置
if ENV == 'production':
    # GCP Cloud SQL 連接
    DB_USER = os.environ.get('DB_USER', 'linkedin_user')
    DB_PASS = os.environ.get('DB_PASS', '')
    DB_NAME = os.environ.get('DB_NAME', 'linkedin_jobs')
    
    # Cloud SQL連接字符串
    # 使用Unix Socket連接到Cloud SQL (適用於Cloud Run)
    INSTANCE_CONNECTION_NAME = os.environ.get('INSTANCE_CONNECTION_NAME', 
                                            'linkedin-scraper-project:asia-east1:linkedin-db')
    DB_HOST = f'/cloudsql/{INSTANCE_CONNECTION_NAME}'
    
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@/{DB_NAME}?host={DB_HOST}"
else:
    # 本地開發環境
    DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@db:5432/linkedin')
```

### 創建雲端專用Dockerfile

創建 `Dockerfile.cloud` 文件：

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    build-essential \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# 安裝Cloud SQL Proxy依賴
RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 設置環境變數
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# 複製應用程序文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 添加Cloud SQL Python連接器
RUN pip install --no-cache-dir google-cloud-secret-manager psycopg2-binary

# 複製應用程序代碼
COPY app/ /app/

# 健康檢查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# 啟動命令
CMD ["python", "-m", "app.main", "--upgrade", "--schedule"]
```

### 創建API端點用於健康檢查

在 `app/api/routes.py` 中添加：

```python
from flask import Flask, jsonify
from app.monitoring.health import get_application_status

app = Flask(__name__)

@app.route('/health')
def health_check():
    """健康檢查端點"""
    status = get_application_status()
    return jsonify(status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

在 `app/main.py` 中添加啟動API服務的程式碼：

```python
import threading
from app.api.routes import app

def start_api_server():
    """啟動API服務器"""
    app.run(host='0.0.0.0', port=8080)

# 在主函數中添加
if args.schedule:
    # 啟動API服務
    api_thread = threading.Thread(target=start_api_server)
    api_thread.daemon = True
    api_thread.start()
    
    # 啟動排程器
    ...
```

### 構建並推送容器

```bash
# 構建容器
docker build -t gcr.io/linkedin-scraper-project/linkedin-scraper:latest -f Dockerfile.cloud .

# 推送到Google Container Registry
gcloud auth configure-docker
docker push gcr.io/linkedin-scraper-project/linkedin-scraper:latest
```

## 部署到Cloud Run

### 部署應用服務

```bash
gcloud run deploy linkedin-scraper \
   --image gcr.io/linkedin-scraper-project/linkedin-scraper:latest \
   --service-account=linkedin-scraper-sa@linkedin-scraper-project.iam.gserviceaccount.com \
   --region=asia-east1 \
   --platform=managed \
   --allow-unauthenticated \
   --memory=1G \
   --cpu=1 \
   --concurrency=1 \
   --max-instances=1 \
   --set-env-vars="DB_USER=linkedin_user,DB_NAME=linkedin_jobs,INSTANCE_CONNECTION_NAME=linkedin-scraper-project:asia-east1:linkedin-db" \
   --add-cloudsql-instances=linkedin-scraper-project:asia-east1:linkedin-db \
   --set-secrets=DB_PASS=linkedin-db-password:latest
```

## 設置定時執行

### 創建Cloud Scheduler任務

```bash
# 獲取服務URL
SERVICE_URL=$(gcloud run services describe linkedin-scraper --platform managed --region asia-east1 --format 'value(status.url)')

# 創建定時任務 (每天中午12點執行)
gcloud scheduler jobs create http linkedin-daily-job-scraper \
   --schedule="0 12 * * *" \
   --uri="${SERVICE_URL}/run-all-configs" \
   --http-method=POST \
   --oidc-service-account-email=linkedin-scraper-sa@linkedin-scraper-project.iam.gserviceaccount.com \
   --oidc-token-audience="${SERVICE_URL}" \
   --time-zone="Asia/Taipei"
```

## 監控和日誌

### 設置Cloud Monitoring告警

```bash
# 創建通知渠道 (email)
gcloud beta monitoring channels create \
   --display-name="LinkedIn Scraper Alerts" \
   --type=email \
   --channel-labels=email_address=your-email@example.com

# 獲取通知渠道ID
CHANNEL_ID=$(gcloud beta monitoring channels list --filter="displayName=LinkedIn Scraper Alerts" --format="value(name)")

# 設置CPU使用率告警
gcloud beta monitoring alerting policies create \
   --display-name="LinkedIn Scraper CPU Alert" \
   --condition-filter="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"linkedin-scraper\" AND metric.type=\"run.googleapis.com/container/cpu/utilization\" AND metric.labels.state=\"used\" AND metric.value>0.8" \
   --condition-threshold-value=0.8 \
   --condition-threshold-duration=60s \
   --notification-channels=${CHANNEL_ID} \
   --combiner=OR
```

### 設置日誌匯出

```bash
# 創建日誌存儲桶
gsutil mb -l asia-east1 gs://linkedin-scraper-logs

# 創建日誌接收器
gcloud logging sinks create linkedin-scraper-logs-sink \
   storage.googleapis.com/linkedin-scraper-logs \
   --log-filter="resource.type=cloud_run_revision AND resource.labels.service_name=linkedin-scraper"

# 獲取分配給接收器的服務帳號
SINK_SERVICE_ACCOUNT=$(gcloud logging sinks describe linkedin-scraper-logs-sink --format="value(writerIdentity)")

# 授予儲存桶寫入權限
gsutil iam ch ${SINK_SERVICE_ACCOUNT}:objectCreator gs://linkedin-scraper-logs
```

### 設置定期備份

```bash
# 創建備份存儲桶
gsutil mb -l asia-east1 gs://linkedin-scraper-backups

# 創建備份腳本 - backup.sh
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d-%H%M%S)
FILENAME="linkedin-db-backup-${DATE}.sql"

# 導出資料庫
gcloud sql export sql linkedin-db gs://linkedin-scraper-backups/${FILENAME} \
   --database=linkedin_jobs \
   --offload

# 刪除超過30天的備份
gsutil rm `gsutil ls gs://linkedin-scraper-backups/ | grep -i linkedin-db-backup | sort | head -n -30` 2>/dev/null || true
EOF

chmod +x backup.sh

# 創建定時備份任務
gcloud scheduler jobs create http linkedin-db-backup \
   --schedule="0 2 * * *" \
   --uri="https://www.googleapis.com/sql/v1beta4/projects/linkedin-scraper-project/instances/linkedin-db/export" \
   --http-method=POST \
   --oauth-service-account-email=linkedin-scraper-sa@linkedin-scraper-project.iam.gserviceaccount.com \
   --oauth-token-scope=https://www.googleapis.com/auth/cloud-platform \
   --headers="Content-Type=application/json" \
   --body="{\"exportContext\":{\"kind\":\"sql#exportContext\",\"fileType\":\"SQL\",\"uri\":\"gs://linkedin-scraper-backups/linkedin-db-backup-\$(date +%Y%m%d-%H%M%S).sql\",\"databases\":[\"linkedin_jobs\"]}}" \
   --time-zone="Asia/Taipei"
```

## 成本與安全考量

### 成本優化

1. **使用合適的資源等級**：
   - 開始使用 `db-f1-micro` 資料庫實例，根據需要擴展
   - 設置 Cloud Run 的 `max-instances=1` 避免過多並行執行

2. **設置預算提醒**：
   ```bash
   gcloud billing budgets create \
      --billing-account=<YOUR_BILLING_ACCOUNT_ID> \
      --display-name="LinkedIn Scraper Budget" \
      --budget-amount=50USD \
      --threshold-rule=percent=80 \
      --threshold-rule=percent=100 \
      --notify-on-cost-basis=forecasted-spend
   ```

3. **使用 Cloud Run 的無服務器特性**：
   - 只在需要時運行爬蟲，避免持續運行成本

### 安全最佳實踐

1. **限制資料庫訪問**：
   ```bash
   # 設置IP授權
   gcloud sql instances patch linkedin-db \
      --authorized-networks=<YOUR_OFFICE_IP>/32
   ```

2. **使用服務帳號和最小權限原則**：
   - 為每個服務組件使用專用服務帳號
   - 只賦予所需的最小權限

3. **加密敏感數據**：
   - 使用 Secret Manager 存儲所有憑證
   - 考慮使用 KMS 加密重要數據

## 故障排除

### 常見問題及解決方案

1. **連接資料庫失敗**
   - 檢查服務帳號是否有 Cloud SQL Client 角色
   - 確認 Cloud SQL 實例名稱和連接字符串格式是否正確
   - 查看 Cloud Run 日誌以獲取更詳細的錯誤信息

2. **爬蟲執行失敗**
   - 檢查 Cloud Run 的容器日誌
   - 確認 Chromium 和 ChromeDriver 是否正確安裝並配置
   - 確保分配了足夠的內存 (建議至少1GB)

3. **定時任務未觸發**
   - 檢查 Cloud Scheduler 日誌
   - 確認時區設置是否正確
   - 驗證服務帳號是否有正確權限

### 查看日誌

```bash
# 查看 Cloud Run 日誌
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=linkedin-scraper" --limit=10

# 查看 Cloud Scheduler 日誌
gcloud logging read "resource.type=cloud_scheduler_job AND resource.labels.job_id=linkedin-daily-job-scraper" --limit=5

# 查看 Cloud SQL 日誌
gcloud logging read "resource.type=cloudsql_database AND resource.labels.database_id=linkedin-db" --limit=5
```

## 結語

本指南提供了將LinkedIn爬蟲專案遷移到GCP的完整步驟。通過遵循這些步驟，您可以構建一個穩健、可擴展且成本效益高的雲端爬蟲系統。定期監控系統性能和成本，並根據需要調整配置以優化資源使用。

對於進一步的問題或支持，請參考[Google Cloud 文檔](https://cloud.google.com/docs)或聯絡GCP支持團隊。 
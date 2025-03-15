FROM python:3.10-slim

# 安裝 Chromium 和相關依賴
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    chromium \
    chromium-driver \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 設置工作目錄
WORKDIR /project

# 複製 requirements.txt 並安裝依賴
COPY requirements.txt /project/
RUN pip install --no-cache-dir -r requirements.txt

# 環境變數設置
ENV PYTHONPATH=/project
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMIUM_PATH=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver
#!/bin/bash

# 設置工作目錄為腳本所在目錄
cd "$(dirname "$0")"

# 顯示啟動信息
echo "======================================"
echo "LinkedIn 職缺爬蟲排程器 - 本地啟動程序"
echo "======================================"
echo "設置為每天中午12點自動執行爬蟲"
echo "======================================"

# 檢查資料庫是否已經初始化
if [ ! -f ".db_initialized" ]; then
    echo "初始化資料庫..."
    python -m app.main --init
    if [ $? -eq 0 ]; then
        touch .db_initialized
        echo "資料庫初始化完成"
    else
        echo "資料庫初始化失敗，請檢查設置"
        exit 1
    fi
fi

# 檢查是否有活躍的搜索配置
echo "檢查搜索配置..."
CONFIGS=$(python -m app.main list-configs 2>&1)
if [[ $CONFIGS == *"沒有找到任何搜索配置"* ]]; then
    echo "未找到任何搜索配置，請先添加至少一個配置"
    echo "可以使用以下命令添加配置："
    echo "python -m app.main add-config --name \"配置名稱\" --keyword \"關鍵字\" --location \"地點\" --max-pages 5"
    exit 1
fi

# 啟動排程器
echo "啟動排程器，設置每天中午12點執行..."
nohup python -m app.main --schedule > scheduler.log 2>&1 &
PID=$!

# 將進程ID保存到檔案中，以便後續操作
echo $PID > scheduler.pid
echo "排程器已啟動，進程ID: $PID"
echo "日誌檔案: scheduler.log"
echo
echo "您可以使用以下命令查看日誌："
echo "tail -f scheduler.log"
echo
echo "如需停止排程器，請執行："
echo "./stop_scheduler.sh" 
#!/bin/bash

# 設置工作目錄為腳本所在目錄
cd "$(dirname "$0")"

# 顯示停止信息
echo "======================================"
echo "LinkedIn 職缺爬蟲排程器 - 停止程序"
echo "======================================"

# 檢查是否存在pid檔案
if [ ! -f "scheduler.pid" ]; then
    echo "找不到排程器進程ID檔案，排程器可能尚未啟動"
    exit 1
fi

# 讀取進程ID
PID=$(cat scheduler.pid)
echo "正在停止進程ID: $PID..."

# 檢查進程是否存在
if ps -p $PID > /dev/null; then
    # 嘗試優雅地停止進程
    kill $PID
    echo "已發送停止信號，等待進程終止..."
    
    # 等待最多10秒
    count=0
    while ps -p $PID > /dev/null && [ $count -lt 10 ]; do
        sleep 1
        ((count++))
    done
    
    # 如果進程仍在運行，強制停止
    if ps -p $PID > /dev/null; then
        echo "進程未能在10秒內終止，正在強制停止..."
        kill -9 $PID
        sleep 1
    fi
    
    # 確認進程是否已停止
    if ps -p $PID > /dev/null; then
        echo "無法停止進程，請手動處理進程ID: $PID"
        exit 1
    else
        echo "排程器已成功停止"
        rm scheduler.pid
    fi
else
    echo "進程ID: $PID 已不存在，可能已經停止"
    rm scheduler.pid
fi

echo "======================================"
echo "如需重新啟動排程器，請執行："
echo "./start_local_scheduler.sh"
echo "======================================" 
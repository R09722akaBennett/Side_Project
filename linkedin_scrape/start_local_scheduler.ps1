# LinkedIn 職缺爬蟲排程器 - Windows PowerShell 啟動腳本

# 設置工作目錄為腳本所在目錄
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $scriptPath

# 顯示啟動信息
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "LinkedIn 職缺爬蟲排程器 - 本地啟動程序 (Windows)" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "設置為每天中午12點自動執行爬蟲" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# 檢查資料庫是否已經初始化
if (-not (Test-Path ".db_initialized")) {
    Write-Host "初始化資料庫..." -ForegroundColor Yellow
    python -m app.main --init
    if ($?) {
        New-Item -Path ".db_initialized" -ItemType File | Out-Null
        Write-Host "資料庫初始化完成" -ForegroundColor Green
    } else {
        Write-Host "資料庫初始化失敗，請檢查設置" -ForegroundColor Red
        exit 1
    }
}

# 檢查是否有活躍的搜索配置
Write-Host "檢查搜索配置..." -ForegroundColor Yellow
$configs = python -m app.main list-configs 2>&1
if ($configs -like "*沒有找到任何搜索配置*") {
    Write-Host "未找到任何搜索配置，請先添加至少一個配置" -ForegroundColor Red
    Write-Host "可以使用以下命令添加配置：" -ForegroundColor Yellow
    Write-Host 'python -m app.main add-config --name "配置名稱" --keyword "關鍵字" --location "地點" --max-pages 5' -ForegroundColor Yellow
    exit 1
}

# 將排程器註冊為 Windows 任務計劃
$action = New-ScheduledTaskAction -Execute "python.exe" -Argument "-m app.main run-all-configs"
$trigger = New-ScheduledTaskTrigger -Daily -At 12:00pm
$settings = New-ScheduledTaskSettingsSet -RunOnlyIfNetworkAvailable -WakeToRun

# 嘗試註冊任務
Write-Host "正在將爬蟲設置為每天中午12點執行的 Windows 排程任務..." -ForegroundColor Yellow
$existingTask = Get-ScheduledTask -TaskName "LinkedInJobScraper" -ErrorAction SilentlyContinue

if ($existingTask) {
    Write-Host "任務已存在，正在更新..." -ForegroundColor Yellow
    Set-ScheduledTask -TaskName "LinkedInJobScraper" -Action $action -Trigger $trigger -Settings $settings | Out-Null
} else {
    Register-ScheduledTask -TaskName "LinkedInJobScraper" -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest | Out-Null
}

# 確認任務是否創建成功
$task = Get-ScheduledTask -TaskName "LinkedInJobScraper" -ErrorAction SilentlyContinue
if ($task) {
    Write-Host "排程任務已成功創建/更新！" -ForegroundColor Green
    Write-Host "任務名稱: LinkedInJobScraper" -ForegroundColor Green
    Write-Host "執行時間: 每天中午12:00" -ForegroundColor Green
    Write-Host "工作目錄: $scriptPath" -ForegroundColor Green
} else {
    Write-Host "排程任務創建失敗，請以管理員身份運行此腳本" -ForegroundColor Red
    exit 1
}

Write-Host "`n要查看已創建的任務，請打開：任務計劃程序 > 任務計劃程序庫 > LinkedInJobScraper" -ForegroundColor Cyan
Write-Host "要停止任務，請使用 Windows 任務計劃程序或執行 stop_scheduler.ps1" -ForegroundColor Cyan 
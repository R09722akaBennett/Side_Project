# LinkedIn 職缺爬蟲排程器 - Windows PowerShell 停止腳本

# 顯示信息
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "LinkedIn 職缺爬蟲排程器 - 停止程序 (Windows)" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# 嘗試停止 Windows 任務計劃中的任務
$taskName = "LinkedInJobScraper"
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue

if ($task) {
    Write-Host "正在停用排程任務: $taskName..." -ForegroundColor Yellow
    try {
        # 停用任務
        Disable-ScheduledTask -TaskName $taskName | Out-Null
        Write-Host "任務已停用" -ForegroundColor Green
        
        # 詢問用戶是否要完全刪除任務
        $confirmation = Read-Host "是否要完全刪除排程任務？(Y/N)"
        if ($confirmation -eq 'Y' -or $confirmation -eq 'y') {
            Unregister-ScheduledTask -TaskName $taskName -Confirm:$false | Out-Null
            Write-Host "排程任務已成功刪除" -ForegroundColor Green
        } else {
            Write-Host "排程任務已保留但被停用，可以在任務計劃程序中重新啟用" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "停用任務時發生錯誤: $_" -ForegroundColor Red
        Write-Host "請嘗試以管理員身份運行此腳本" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "找不到名為 $taskName 的排程任務" -ForegroundColor Red
    exit 1
}

# 檢查是否有正在運行的 Python 進程
$pythonProcesses = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*app.main*" }

if ($pythonProcesses) {
    Write-Host "發現正在運行的 Python 進程，嘗試停止..." -ForegroundColor Yellow
    foreach ($process in $pythonProcesses) {
        try {
            $process | Stop-Process -Force
            Write-Host "進程 ID $($process.Id) 已停止" -ForegroundColor Green
        } catch {
            Write-Host "無法停止進程 ID $($process.Id): $_" -ForegroundColor Red
        }
    }
}

Write-Host "`n排程器已停止。如需重新啟動，請執行 start_local_scheduler.ps1" -ForegroundColor Cyan 
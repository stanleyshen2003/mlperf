# Fake model server: start / stop via Make
# Usage: make start [PORT=8000]  |  make stop  |  make restart  |  make status
# Uses PowerShell on Windows for start/stop/status.

PORT ?= 8000
PIDFILE := .server.pid
SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -Command

.PHONY: start stop restart status workload workload-schedule loadgen-workload kserve-schedule

start:
	@if (Test-Path '$(PIDFILE)') { $$p = (Get-Content '$(PIDFILE)' -Raw).Trim(); try { Get-Process -Id $$p -ErrorAction Stop | Out-Null; Write-Host 'Server already running (PID' $$p '). Use make stop first.'; exit 1 } catch {}; Remove-Item '$(PIDFILE)' -Force }; $$proc = Start-Process -FilePath 'uv' -ArgumentList 'run','python','run.py','--host','127.0.0.1','--port','$(PORT)' -WorkingDirectory 'fakeserver' -PassThru -NoNewWindow; $$proc.Id | Set-Content '$(PIDFILE)'; Write-Host 'Server started (PID' $$proc.Id '). http://127.0.0.1:$(PORT)'; Start-Sleep -Seconds 1; if (Test-Path '$(PIDFILE)') { $$p = (Get-Content '$(PIDFILE)' -Raw).Trim(); try { Get-Process -Id $$p -ErrorAction Stop | Out-Null; Write-Host 'Server running (PID' $$p '). Port $(PORT).'; try { (Invoke-WebRequest -Uri 'http://127.0.0.1:$(PORT)/health/' -UseBasicParsing -TimeoutSec 2).Content } catch {} } catch { Write-Host 'PID file exists but process not running.' } } else { Write-Host 'Server not running.' }

stop:
	@if (Test-Path '$(PIDFILE)') { $$p = (Get-Content '$(PIDFILE)' -Raw).Trim(); try { Stop-Process -Id $$p -Force -ErrorAction Stop; Write-Host 'Server stopped (PID' $$p ').' } catch { Write-Host 'Process' $$p 'not running.' }; Remove-Item '$(PIDFILE)' -Force -ErrorAction SilentlyContinue } else { Write-Host 'No $(PIDFILE) found. Server may not be running.' }

restart: stop
	@Start-Sleep -Seconds 1; cmd /c "make start"

status:
	@if (Test-Path '$(PIDFILE)') { $$p = (Get-Content '$(PIDFILE)' -Raw).Trim(); try { Get-Process -Id $$p -ErrorAction Stop | Out-Null; Write-Host 'Server running (PID' $$p '). Port $(PORT).'; try { (Invoke-WebRequest -Uri 'http://127.0.0.1:$(PORT)/health/' -UseBasicParsing -TimeoutSec 2).Content } catch {} } catch { Write-Host 'PID file exists but process not running. Run make stop to clean.' } } else { Write-Host 'Server not running (no $(PIDFILE)).' }

workload:
	uv run python custom/workload.py --url http://127.0.0.1:$(PORT) --duration 10 --concurrency 6 --mix 1,1,1

workload-schedule:
	uv run python custom/loadgen_workload.py --url http://127.0.0.1:$(PORT) --schedule "0-5:3,5-10:10,10-15:20" --workload-log custom/workload_schedule.csv

loadgen-workload:
	uv run python custom/loadgen_workload.py --url http://127.0.0.1:$(PORT) --duration-ms 10000 --target-qps 10 --mix 1,1,1 --workload-log custom/workload.csv

KSERVE_URL ?= http://140.113.194.247:8080
kserve-schedule:
	uv run python custom/loadgen_workload.py --kserve --url $(KSERVE_URL) --schedule "0-5:3,5-10:10,10-15:20,15-25:40" --workload-log custom/workload_kserve_schedule.csv

@echo off
REM ===========================================================================
REM VivaSense exam-station band relay.
REM
REM Holds the Bluetooth link to the heart-rate band and feeds whichever
REM physical session is running. Started once, at boot, by Windows Task
REM Scheduler - see install_station_relay.bat. Nobody types anything.
REM
REM Edit the two settings below for this station, then run
REM install_station_relay.bat ONCE as Administrator.
REM ===========================================================================

REM --- The API root. NOT a session URL: the relay finds the session itself.
REM     This MUST point at the same backend the kiosk browser is talking to.
REM     Getting it wrong is silent from the band's side: the OLED shows a
REM     pulse, the relay stays connected, and nothing ever reaches the system.
REM       kiosk on https://www.vivasense.tech -> https://api.vivasense.tech/api
REM       Django running on THIS machine      -> http://127.0.0.1:8000/api
set BACKEND=https://api.vivasense.tech/api

REM --- Must match EXAM_STATION_TOKEN in the backend's environment.
set STATION_TOKEN=2208720c-f09f-4fa7-8070-7663ea807d605e3d55e9-7658-4285-81ba-c12a6bacd46c

REM --- Advertised name of the band. Leave as-is unless it was renamed.
set BAND_NAME=VivaSense-HR

REM ---------------------------------------------------------------------------
cd /d "%~dp0..\.."

if not exist "venv\Scripts\python.exe" (
  echo [relay] venv\Scripts\python.exe not found. Run this from the backend folder.
  timeout /t 20 >nul
  exit /b 1
)

if "%STATION_TOKEN%"=="CHANGE_ME" (
  echo [relay] STATION_TOKEN is not set. Edit run_station_relay.bat first.
  timeout /t 20 >nul
  exit /b 1
)

REM The relay reconnects internally, but if the process itself ever dies -
REM a Bluetooth stack reset, a driver reload - bring it straight back.
:loop
echo [relay] starting %DATE% %TIME%
venv\Scripts\python.exe -m physiology.station_sidecar ^
    --backend %BACKEND% ^
    --token %STATION_TOKEN% ^
    --device %BAND_NAME%
echo [relay] exited; restarting in 10s
timeout /t 10 >nul
goto loop

@echo off
REM ===========================================================================
REM Install the exam-station band relay as a Windows scheduled task.
REM
REM Run ONCE per exam station, as Administrator. The token is already filled
REM into run_station_relay.bat; check BACKEND there points at the right server.
REM
REM After this the relay starts at every logon and keeps itself alive. No
REM terminal, no command to remember, nothing for an invigilator to do.
REM
REM This file only handles elevation; the registration itself lives in
REM install_station_relay.ps1, because several of the defaults schtasks
REM applies are wrong for a machine that must run unattended.
REM ===========================================================================

setlocal

net session >nul 2>&1
if errorlevel 1 (
  echo Right-click this file and choose "Run as administrator".
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_station_relay.ps1"
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
  echo.
  echo Installation failed.
  pause
  exit /b %RC%
)

echo.
echo Done. The relay starts whenever you log in, on battery or mains.
echo.
echo   check    schtasks /Query /TN "VivaSenseStationRelay"
echo   stop     schtasks /End   /TN "VivaSenseStationRelay"
echo   remove   schtasks /Delete /TN "VivaSenseStationRelay" /F
echo.
pause
endlocal

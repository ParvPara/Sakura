@echo off
echo Starting Sakura LLM System...

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    echo Make sure you're in the correct directory
    pause
    exit /b 1
)

:: Kill existing processes
echo Killing existing processes...
taskkill /F /IM python.exe 2>nul
taskkill /F /IM cloudflared.exe 2>nul

:: Wait
timeout /t 3 /nobreak >nul

:: Start LLM controller first
echo Starting LLM Controller API...
start "" cmd /k "venv\Scripts\activate && python llm_controller.py"

:: Wait for controller to start
echo Waiting for controller to start...
timeout /t 8 /nobreak >nul

:: Test if controller is running
echo Testing LLM Controller...
curl -s http://localhost:4000/status >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ LLM Controller is running
) else (
    echo ❌ LLM Controller failed to start
    echo Please check the LLM Controller window for errors
    pause
    exit /b 1
)

:: Start Discord bot
echo Starting Discord Bot...
start "" cmd /k "venv\Scripts\activate && python main.py"

:: Wait for bot to fully initialize and register with controller
echo Waiting for bot to fully initialize...
echo Checking for bot connection every 10 seconds...

:wait_for_bot
timeout /t 10 /nobreak >nul
echo Checking bot status...

:: Get the status and save to temp file for debugging
curl -s "http://localhost:4000/status" > temp_status.json 2>nul

:: Check if servers are available (which means bot is connected with guild data)
curl -s "http://localhost:4000/servers" > temp_servers.json 2>nul
findstr /C:"\"servers\":\[{" temp_servers.json >nul 2>&1
if %errorlevel% neq 0 (
    echo Status check: Bot not yet connected or no guild data available
    type temp_status.json | findstr "bot_connected"
    echo Still waiting for bot to connect and register...
    del temp_servers.json >nul 2>&1
    goto wait_for_bot
)

echo ✅ Bot is fully connected and registered!
echo Server data available:
type temp_servers.json

:: Clean up temp files
del temp_status.json >nul 2>&1
del temp_servers.json >nul 2>&1

:: Additional small wait to ensure guild data is sent
timeout /t 5 /nobreak >nul

:: Start LLM Controller tunnel
echo Starting LLM Controller Tunnel...
start "LLM Controller Tunnel" .\cloudflared.exe tunnel --config config\llm_tunnel.yml run

:: Start Discord Bot tunnel
echo Starting Discord Bot Tunnel...
start "Discord Bot Tunnel" .\cloudflared.exe tunnel --config config\discord_tunnel.yml run

:: Wait for tunnels to start
timeout /t 8 /nobreak >nul

echo.
echo All processes started!
echo.
echo Testing final status...
timeout /t 2 /nobreak >nul

:: Test API status
echo Testing API status...
curl -s http://localhost:4000/status
echo.

echo.
echo 🌸 Sakura LLM System is running! 🌸
echo.
echo LLM Controller API: http://localhost:4000
echo LLM Controller Tunnel: https://llm-api.your-domain.com
echo Discord Bot API: http://localhost:5000
echo Discord Bot Tunnel: https://api.your-domain.com
echo.
echo Check the bot window to see if it connected to the controller.
echo.
pause

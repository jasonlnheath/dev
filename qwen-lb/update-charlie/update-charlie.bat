@echo off
echo ============================================
echo   Charlie Hermes Agent Update Script
echo ============================================
echo.

REM --- Step 1: Pull latest Hermes from upstream ---
echo [1/5] Pulling latest Hermes agent...
wsl -d Ubuntu -- bash -c "cd ~/.hermes/hermes-agent && git stash && git pull origin main && git stash pop 2>/dev/null; echo done"
echo.

REM --- Step 2: Install custom provider plugin (thinking control) ---
echo [2/5] Installing custom provider plugin...
wsl -d Ubuntu -- bash -c "mkdir -p ~/.hermes/hermes-agent/plugins/model-providers/custom && cp /mnt/c/dev/qwen-lb/update-charlie/custom_provider/__init__.py ~/.hermes/hermes-agent/plugins/model-providers/custom/__init__.py && echo 'Custom provider installed'"
echo.

REM --- Step 3: Install PriorityCompressor plugin ---
echo [3/5] Installing PriorityCompressor plugin...
wsl -d Ubuntu -- bash -c "mkdir -p ~/.hermes/hermes-agent/plugins/context_engine/priority_compressor && cp /mnt/c/dev/qwen-lb/update-charlie/priority_compressor/__init__.py ~/.hermes/hermes-agent/plugins/context_engine/priority_compressor/__init__.py && cp /mnt/c/dev/qwen-lb/update-charlie/priority_compressor/plugin.yaml ~/.hermes/hermes-agent/plugins/context_engine/priority_compressor/plugin.yaml && echo 'PriorityCompressor installed'"
echo.

REM --- Step 4: Patch config.yaml ---
echo [4/5] Patching config.yaml (model.max_tokens, compression, context engine)...
wsl -d Ubuntu -- bash -c "python3 /mnt/c/dev/qwen-lb/update-charlie/config-patch.py ~/.hermes/config.yaml"
echo.

REM --- Step 5: Restart gateway ---
echo [5/5] Restarting Hermes gateway...
wsl -d Ubuntu -- bash -c "systemctl --user restart hermes-gateway && sleep 3 && systemctl --user status hermes-gateway | head -5"
echo.

echo ============================================
echo   Update complete!
echo ============================================
echo.
echo Changes applied:
echo   - Custom provider: thinking capped at 2K, disabled for sub-agents
echo   - PriorityCompressor context engine installed
echo   - model.max_tokens set to 32768 (fixes write_file truncation)
echo   - Compression enabled
echo.
pause

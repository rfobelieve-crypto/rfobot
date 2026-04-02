@echo off
chcp 65001 >nul 2>&1
cd /d "C:\Users\rfo\Desktop\flowbot\flow_system"
python -m indicator.auto_update --once

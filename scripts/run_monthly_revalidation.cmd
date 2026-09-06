@echo off
REM Monthly v7 edge re-validation ritual (scheduled: 5th of each month).
REM Runs the quarterly_revalidation script locally so it has .env + parquet +
REM network + MySQL. Writes a dated report and pushes the verdict to Telegram.
cd /d "C:\Users\rfo\Desktop\flowbot\flow_system"
"C:\Users\rfo\anaconda3\python.exe" -m research.dual_model.quarterly_revalidation >> "research\results\dual_model\_revalidation_cron.log" 2>&1

"C:\Usersfonaconda3\python.exe" research\sweep_failure\sweep_forward.py >> "researchesults\dual_model\_revalidation_cron.log" 2>&1

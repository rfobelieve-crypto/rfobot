@echo off
REM Discord watcher for the HMM live engine (strategy line 4, ../arb repo).
REM Watcher: research/ops/hmm_watch.py  (transition-only: silent when fine)
REM
REM NO --pair ON PURPOSE. The target is derived from the watchdog's $Members
REM minus logs/stop/*.stop, via research/ops/live_hmm.py -- the same function
REM freshness_board reads. On 2026-09-14 a hardcoded target burned us four
REM times in one day: AERO was stopped but its watcher was not, so Discord
REM got a false alarm every five minutes all afternoon, and the real red
REM light (scan_pull dead 31 times) was buried under it.
REM
REM ASCII ONLY. cmd.exe reads .bat in the OEM codepage (cp950 here); UTF-8
REM CJK in comments shifts its byte-offset bookkeeping and it SKIPS LINES
REM while still exiting 0 (mistake.md 2026-09-13: six recorders silently
REM never started). CRLF line endings for the same reason.
REM
REM Launched by research\ops\run_hidden.vbs (wscript.exe has no console, so
REM nothing pops up every five minutes). Proof it ran is the ARTIFACT: new
REM lines in research\results\hmm_watch.log. Never LastTaskResult -- an
REM async launcher's exit code says nothing about the child process
REM (mistake.md 2026-09-06).
REM
REM Replaces a background shell loop that the harness killed twice in one
REM evening under memory pressure. A live-money engine's safety net must not
REM depend on a chat session staying alive.
setlocal
set ROOT=C:\Users\rfo\Desktop\flowbot\flow_system
set PYTHONIOENCODING=utf-8
cd /d "%ROOT%"
python "%ROOT%\research\ops\hmm_watch.py" >> "%ROOT%\research\results\hmm_watch.log" 2>&1
endlocal

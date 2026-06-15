import re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
files = [ROOT/"indicator"/"dashboard.py"] + sorted((ROOT/"indicator"/"dashboard_tabs").glob("*.py"))
# brighten the Nansen green: #34e0a0 (rgb 52,224,160) -> #36ffae (rgb 54,255,174)
n=0
for f in files:
    s=f.read_text(encoding="utf-8"); b=s
    s=re.sub("#34e0a0","#36ffae",s,flags=re.IGNORECASE)
    s=re.sub(r"52,\s*224,\s*160","54,255,174",s)
    if s!=b: f.write_text(s,encoding="utf-8"); n+=1
print("brightened green in",n,"files")
# verify no old green left
left=sum(len(re.findall(r"34e0a0|52,\s*224,\s*160",f.read_text(encoding='utf-8'),re.I)) for f in files)
print("old-green leftovers (0 expected):",left)

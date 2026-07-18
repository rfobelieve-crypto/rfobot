"""Remote MCP entry — streamable-HTTP transport for claude.ai Connectors.

TODO 4.5 (2026-07-17 登記): 把唯讀 agent 工具開放給所有 Claude 介面
(手機 app / claude.ai / Claude Code)，不再只有本機 stdio 的 Claude
Desktop 摸得到。使用者日常 = 問一句「看一下撤單流」，Claude 自己呼叫
analyze_cancel_flow 等工具並用人話判讀。

Security (fail-closed, 比照 ADMIN_HEAL_TOKEN 模式):
  · AGENT_MCP_TOKEN 未設或太短 → 拒絕啟動 (絕不裸奔)
  · token 內嵌 URL 路徑 (claude.ai custom connector 無 OAuth 時
    不能帶自訂 header): /<token>/mcp — 路徑錯 = 404, 什麼都不服務
  · agent-boundary 規則全部照舊: 唯讀工具、禁 executor/inference
    imports (tests/test_agent_boundary.py AST 守衛涵蓋本檔)

Run (獨立 Railway service, Dockerfile.indicator image):
    python -m indicator.agent.http_server
Connector URL:
    https://<service-domain>/<AGENT_MCP_TOKEN>/mcp
"""
from __future__ import annotations

import os
import sys

from mcp.server.transport_security import TransportSecuritySettings

from indicator.agent.server import mcp


def main() -> int:
    token = os.getenv("AGENT_MCP_TOKEN", "").strip()
    if len(token) < 16:
        print("FATAL: AGENT_MCP_TOKEN missing or <16 chars — refusing to "
              "start (fail-closed; set it in the service variables)",
              file=sys.stderr)
        return 1
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = int(os.getenv("PORT", "8080"))
    mcp.settings.streamable_http_path = f"/{token}/mcp"
    # DNS-rebinding Host 檢查是本機 server 的防線;公網 HTTPS + 路徑 token
    # 下反而把 Railway domain 擋成 421 — 關閉(auth 邊界在 token 路徑)
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False)
    print(f"rfobot-orderflow MCP (streamable-http) on "
          f":{mcp.settings.port}/<token>/mcp", file=sys.stderr)
    mcp.run(transport="streamable-http")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

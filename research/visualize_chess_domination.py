"""
ChessDomination Pressure Framework — 3D 可視化模組
===================================================
使用 Plotly 建立三維棋局沙盤，含 CUFD 顏色映射和策略邊界面。

使用方式：
    python research/visualize_chess_domination.py

    或在 notebook 中：
        from research.visualize_chess_domination import plot_3d_chessboard
        fig = plot_3d_chessboard(df)
        fig.show()
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 確保可以 import 專案模組
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.features.chess_domination import (
    CDPConfig,
    CDPResult,
    ChessDominationPressure,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 3D 散點圖（主要可視化）
# ═══════════════════════════════════════════════════════════════════════════

def plot_3d_chessboard(
    df: pd.DataFrame,
    result: Optional[CDPResult] = None,
    config: Optional[CDPConfig] = None,
    title: str = "三維棋局沙盤 · H1 實時",
    show_boundaries: bool = True,
    marker_size_range: tuple = (3, 18),
    last_n_bars: Optional[int] = None,
):
    """
    建立 3D Scatter Plot，展示 XYZ 三維空間 + CUFD 顏色映射。

    Parameters:
        df: 含有 OHLCV + Coinglass 的 1h DataFrame
        result: 預計算的 CDPResult（None 則自動計算）
        config: CDP 參數配置
        title: 圖表標題
        show_boundaries: 是否顯示策略邊界面（Y=0 楚河, Z=0.5 重炮門檻）
        marker_size_range: 散點大小範圍 (min, max)
        last_n_bars: 只顯示最近 N 根 bar（None = 全部）

    Returns:
        plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("需要 plotly：pip install plotly")

    # 計算 CDP（如果沒有預計算結果）
    if result is None:
        engine = ChessDominationPressure(config)
        result = engine.compute_all(df)

    # 準備資料
    x, y, z = result.x, result.y, result.z
    cufd, cdp = result.cufd, result.cdp
    zones = result.zone

    if last_n_bars:
        x = x.iloc[-last_n_bars:]
        y = y.iloc[-last_n_bars:]
        z = z.iloc[-last_n_bars:]
        cufd = cufd.iloc[-last_n_bars:]
        cdp = cdp.iloc[-last_n_bars:]
        zones = zones.iloc[-last_n_bars:]

    # Marker size = 空間勢能 (X² + Z²) → 歸一化到 size_range
    energy = x ** 2 + z ** 2
    e_min, e_max = energy.min(), energy.max()
    if e_max > e_min:
        size_norm = (energy - e_min) / (e_max - e_min)
    else:
        size_norm = pd.Series(0.5, index=energy.index)
    sizes = marker_size_range[0] + size_norm * (marker_size_range[1] - marker_size_range[0])

    # Hover text
    hover = []
    for i in range(len(x)):
        ts = x.index[i].strftime("%Y-%m-%d %H:%M") if hasattr(x.index[i], "strftime") else str(x.index[i])
        hover.append(
            f"<b>{ts}</b><br>"
            f"X(收益區間): {x.iloc[i]:.3f}<br>"
            f"Y(資金流向): {y.iloc[i]:.3f}<br>"
            f"Z(大單強度): {z.iloc[i]:.3f}<br>"
            f"CUFD(累積壓力): {cufd.iloc[i]:.6f}<br>"
            f"CDP: {cdp.iloc[i]:.4f}<br>"
            f"Zone: {zones.iloc[i]}"
        )

    # ── 主散點圖 ──
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x.values,
        y=y.values,
        z=z.values,
        mode="markers",
        marker=dict(
            size=sizes.values,
            color=cufd.values,
            colorscale=[
                [0.0, "rgb(220,220,255)"],     # 低壓力：淡藍
                [0.3, "rgb(255,255,150)"],     # 中低壓力：淡黃
                [0.5, "rgb(255,200,50)"],      # 中壓力：橙黃
                [0.7, "rgb(255,100,50)"],      # 高壓力：橙紅
                [1.0, "rgb(180,0,0)"],         # 極高壓力：深紅
            ],
            colorbar=dict(
                title=dict(text="CUFD<br>累積壓力", font=dict(size=12)),
                thickness=20,
                len=0.7,
            ),
            opacity=0.8,
            line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
        ),
        text=hover,
        hoverinfo="text",
        name="H1 Bars",
    ))

    # ── 最新一根 bar 用星號標記 ──
    fig.add_trace(go.Scatter3d(
        x=[x.iloc[-1]],
        y=[y.iloc[-1]],
        z=[z.iloc[-1]],
        mode="markers+text",
        marker=dict(
            size=15,
            color="yellow",
            symbol="diamond",
            line=dict(width=2, color="black"),
        ),
        text=[f"NOW: {zones.iloc[-1]}"],
        textposition="top center",
        textfont=dict(size=10, color="white"),
        name="當前位置",
        showlegend=True,
    ))

    # ── 策略邊界面 ──
    if show_boundaries:
        # Y = 0 楚河漢界面（半透明藍色）
        x_range = np.linspace(-1, 1, 10)
        z_range = np.linspace(0, 1, 10)
        x_grid, z_grid = np.meshgrid(x_range, z_range)
        y_zero = np.zeros_like(x_grid)

        fig.add_trace(go.Surface(
            x=x_grid, y=y_zero, z=z_grid,
            colorscale=[[0, "rgba(100,150,255,0.15)"], [1, "rgba(100,150,255,0.15)"]],
            showscale=False,
            name="楚河漢界 (Y=0)",
            hoverinfo="name",
        ))

        # Z = 0.5 重炮門檻面（半透明紅色）
        x_range2 = np.linspace(-1, 1, 10)
        y_range2 = np.linspace(-1, 1, 10)
        x_grid2, y_grid2 = np.meshgrid(x_range2, y_range2)
        z_half = np.full_like(x_grid2, 0.5)

        fig.add_trace(go.Surface(
            x=x_grid2, y=y_grid2, z=z_half,
            colorscale=[[0, "rgba(255,100,100,0.12)"], [1, "rgba(255,100,100,0.12)"]],
            showscale=False,
            name="重炮門檻 (Z=0.5)",
            hoverinfo="name",
        ))

    # ── 佈局 ──
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=18),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(
                title="X 收益區間 (價格位置)",
                range=[-1.1, 1.1],
                gridcolor="rgba(200,200,200,0.3)",
            ),
            yaxis=dict(
                title="Y 主力資金流向 (楚河歸誰屬)",
                range=[-1.1, 1.1],
                gridcolor="rgba(200,200,200,0.3)",
            ),
            zaxis=dict(
                title="Z 巨鲸大單強度 (重炮火力)",
                range=[-0.05, 1.05],
                gridcolor="rgba(200,200,200,0.3)",
            ),
            bgcolor="rgb(15,15,25)",
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=0.9),
            ),
        ),
        paper_bgcolor="rgb(10,10,20)",
        font=dict(color="white"),
        legend=dict(
            bgcolor="rgba(30,30,50,0.8)",
            font=dict(size=11),
        ),
        width=1000,
        height=750,
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CUFD 時序圖（輔助可視化）
# ═══════════════════════════════════════════════════════════════════════════

def plot_cufd_timeline(
    df: pd.DataFrame,
    result: Optional[CDPResult] = None,
    config: Optional[CDPConfig] = None,
    title: str = "CUFD 因果累積壓力時序圖",
):
    """
    繪製 CUFD 時序圖 + θ 閾值線 + 過度延伸區域高亮。

    Returns:
        plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("需要 plotly：pip install plotly")

    if result is None:
        engine = ChessDominationPressure(config)
        result = engine.compute_all(df)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["CUFD 因果累積壓力", "CDP 濃縮特徵", "棋局區域"],
        row_heights=[0.4, 0.35, 0.25],
    )

    idx = result.cufd.index

    # ── Panel 1: CUFD ──
    fig.add_trace(go.Scatter(
        x=idx, y=result.cufd,
        mode="lines",
        line=dict(color="rgb(255,80,80)", width=1.5),
        name="CUFD",
        fill="tozeroy",
        fillcolor="rgba(255,80,80,0.15)",
    ), row=1, col=1)

    # θ 閾值線
    fig.add_hline(
        y=result.theta, row=1, col=1,
        line=dict(color="yellow", width=1, dash="dash"),
        annotation=dict(text=f"θ={result.theta:.5f}", font=dict(color="yellow", size=10)),
    )

    # ── Panel 2: CDP ──
    cdp = result.cdp
    colors = np.where(cdp > 0, "rgba(255,80,80,0.6)", "rgba(80,200,80,0.6)")
    fig.add_trace(go.Bar(
        x=idx, y=cdp,
        marker_color=colors.tolist(),
        name="CDP Score",
    ), row=2, col=1)

    # ── Panel 3: Zone encoding ──
    zone_map = {
        "火力枯竭區": -2, "空方過度延伸危險區": -5, "空方壓境優勢區": -4,
        "空方輕度優勢區": -1, "楚河漢界": 0, "中性區": 0,
        "紅方輕度優勢區": 1, "紅方壓境優勢區": 4, "紅方過度延伸危險區": 5,
    }
    zone_vals = result.zone.map(zone_map).fillna(0)
    zone_colors = np.where(
        zone_vals > 3, "rgba(255,50,50,0.8)",
        np.where(zone_vals < -3, "rgba(50,200,50,0.8)",
                 np.where(zone_vals > 0, "rgba(255,150,100,0.6)",
                          np.where(zone_vals < 0, "rgba(100,200,150,0.6)",
                                   "rgba(150,150,150,0.4)")))
    )
    fig.add_trace(go.Bar(
        x=idx, y=zone_vals,
        marker_color=zone_colors.tolist(),
        name="Zone",
        hovertext=result.zone.values,
    ), row=3, col=1)

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16), x=0.5),
        paper_bgcolor="rgb(10,10,20)",
        plot_bgcolor="rgb(15,15,25)",
        font=dict(color="white"),
        showlegend=True,
        legend=dict(bgcolor="rgba(30,30,50,0.8)"),
        height=800,
        width=1200,
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 四維儀表板（X/Y/Z/CUFD 面板總覽）
# ═══════════════════════════════════════════════════════════════════════════

def plot_4d_dashboard(
    df: pd.DataFrame,
    result: Optional[CDPResult] = None,
    config: Optional[CDPConfig] = None,
    title: str = "四維因果棋局儀表板 · H1",
):
    """
    4 面板儀表板：X / Y / Z / CUFD 各自的時序圖 + CDP 疊加。

    Returns:
        plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("需要 plotly：pip install plotly")

    if result is None:
        engine = ChessDominationPressure(config)
        result = engine.compute_all(df)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            "X — 收益區間（圓圈壓境處）",
            "Y — 主力資金流向（楚河歸誰屬）",
            "Z — 巨鲸大單強度（重炮護糧道）",
            "CUFD — 因果累積壓力（第四維）",
        ],
        row_heights=[0.25, 0.25, 0.25, 0.25],
    )

    idx = result.x.index

    # Panel 1: X
    fig.add_trace(go.Scatter(
        x=idx, y=result.x, mode="lines",
        line=dict(color="cyan", width=1.2), name="X",
    ), row=1, col=1)
    fig.add_hline(y=0, row=1, col=1, line=dict(color="gray", dash="dot", width=0.5))

    # Panel 2: Y
    y_colors = np.where(result.y > 0, "rgba(255,100,100,0.7)", "rgba(100,255,100,0.7)")
    fig.add_trace(go.Bar(
        x=idx, y=result.y, marker_color=y_colors.tolist(), name="Y",
    ), row=2, col=1)
    fig.add_hline(y=0, row=2, col=1, line=dict(color="white", dash="dot", width=0.8))
    fig.add_hline(y=0.7, row=2, col=1, line=dict(color="red", dash="dash", width=0.5))
    fig.add_hline(y=-0.7, row=2, col=1, line=dict(color="green", dash="dash", width=0.5))

    # Panel 3: Z
    fig.add_trace(go.Scatter(
        x=idx, y=result.z, mode="lines",
        line=dict(color="orange", width=1.2), name="Z",
        fill="tozeroy", fillcolor="rgba(255,165,0,0.15)",
    ), row=3, col=1)
    fig.add_hline(y=0.5, row=3, col=1, line=dict(color="red", dash="dash", width=0.8))
    fig.add_hline(y=0.15, row=3, col=1, line=dict(color="gray", dash="dot", width=0.5))

    # Panel 4: CUFD
    fig.add_trace(go.Scatter(
        x=idx, y=result.cufd, mode="lines",
        line=dict(color="rgb(255,80,80)", width=1.5), name="CUFD",
        fill="tozeroy", fillcolor="rgba(255,80,80,0.15)",
    ), row=4, col=1)
    fig.add_hline(
        y=result.theta, row=4, col=1,
        line=dict(color="yellow", dash="dash", width=1),
        annotation=dict(text=f"θ={result.theta:.5f}", font=dict(color="yellow", size=10)),
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16), x=0.5),
        paper_bgcolor="rgb(10,10,20)",
        plot_bgcolor="rgb(15,15,25)",
        font=dict(color="white"),
        showlegend=False,
        height=1000,
        width=1200,
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    快速測試：使用 Binance API 抓取即時資料並顯示 3D 沙盤。

    用法：python research/visualize_chess_domination.py [--bars 200] [--save output.html]
    """
    import argparse

    parser = argparse.ArgumentParser(description="CDP 3D Chessboard Visualization")
    parser.add_argument("--bars", type=int, default=200, help="顯示最近 N 根 bar")
    parser.add_argument("--save", type=str, default=None, help="儲存 HTML 到指定路徑")
    args = parser.parse_args()

    # 嘗試從 data_fetcher 取得即時資料
    try:
        from indicator.data_fetcher import fetch_binance_klines, fetch_coinglass
        from indicator.feature_builder_live import build_live_features

        print("正在取得 Binance + Coinglass 資料...")
        klines = fetch_binance_klines(limit=500)
        cg_data = {}
        for ep in ["oi", "funding", "taker", "long_short", "global_ls",
                    "liquidation", "futures_cvd_agg"]:
            try:
                cg_data[ep] = fetch_coinglass(ep, "BTC", interval="1h", limit=500)
            except Exception as e:
                print(f"  跳過 {ep}: {e}")

        print("正在建構特徵...")
        df = build_live_features(klines, cg_data)

    except ImportError:
        print("無法 import data_fetcher，使用模擬資料...")
        # 模擬資料
        np.random.seed(42)
        n = 500
        idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
        close = 85000 + np.cumsum(np.random.randn(n) * 100)
        df = pd.DataFrame({
            "open": close - np.random.rand(n) * 50,
            "high": close + np.random.rand(n) * 100,
            "low": close - np.random.rand(n) * 100,
            "close": close,
            "volume": np.random.lognormal(8, 0.5, n),
            "taker_buy_vol": np.random.lognormal(7.5, 0.5, n),
            "trade_count": np.random.randint(5000, 50000, n),
            "cg_funding_close": np.random.randn(n) * 0.0003,
            "cg_oi_close": 15e9 + np.cumsum(np.random.randn(n) * 1e8),
            "cg_taker_buy": np.random.lognormal(20, 0.3, n),
            "cg_taker_sell": np.random.lognormal(20, 0.3, n),
        }, index=idx)
        df["cg_taker_delta"] = df["cg_taker_buy"] - df["cg_taker_sell"]

    print("正在計算 CDP 框架...")
    engine = ChessDominationPressure()
    result = engine.compute_all(df)

    # 印出最新狀態
    summary = result.summary(-1)
    print("\n" + "=" * 60)
    print("  CDP 四維因果棋局 · 最新狀態")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:>18s}: {v}")
    print("=" * 60)

    # 繪圖
    print("\n正在繪製 3D 沙盤...")
    fig_3d = plot_3d_chessboard(df, result, last_n_bars=args.bars)

    if args.save:
        fig_3d.write_html(args.save)
        print(f"已儲存到: {args.save}")
    else:
        fig_3d.show()

    print("\n正在繪製 CUFD 時序圖...")
    fig_cufd = plot_cufd_timeline(df, result)
    fig_cufd.show()

    print("\n正在繪製四維儀表板...")
    fig_4d = plot_4d_dashboard(df, result)
    fig_4d.show()

    print("\n完成！")

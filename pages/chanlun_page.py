# chanlun_page.py — 缠论自动分析页面
# 自包含：所有缠论算法与UI代码均在本文件中
# 依赖：streamlit, pandas, numpy, yfinance, akshare, streamlit-lightweight-charts

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
from streamlit_lightweight_charts import renderLightweightCharts

# ============================================================
#  数据类型定义
# ============================================================

class FractalType(Enum):
    TOP = "顶分型"
    BOTTOM = "底分型"

class BiDirection(Enum):
    UP = "up"
    DOWN = "down"

class BSPointType(Enum):
    BUY_1 = "一买"
    BUY_2 = "二买"
    BUY_3 = "三买"
    SELL_1 = "一卖"
    SELL_2 = "二卖"
    SELL_3 = "三卖"

@dataclass
class ChanKLine:
    """合并处理后的缠论K线"""
    idx: int
    date: str
    high: float
    low: float
    open: float
    close: float
    is_merged: bool = False
    merged_from: list = field(default_factory=list)

@dataclass
class Fractal:
    """分型"""
    type: FractalType
    chan_kline: ChanKLine
    value: float
    idx: int

@dataclass
class Bi:
    """笔"""
    direction: BiDirection
    start_fractal: Fractal
    end_fractal: Fractal
    start_idx: int
    end_idx: int
    start_value: float
    end_value: float
    start_date: str = ""
    end_date: str = ""

@dataclass
class Segment:
    """线段"""
    direction: BiDirection
    start_idx: int
    end_idx: int
    start_value: float
    end_value: float
    start_date: str = ""
    end_date: str = ""
    bi_count: int = 0

@dataclass
class ZhongShu:
    """中枢"""
    zg: float
    zd: float
    gg: float
    dd: float
    start_idx: int
    end_idx: int
    start_date: str = ""
    end_date: str = ""
    segments: list = field(default_factory=list)

@dataclass
class BSPoint:
    """买卖点"""
    type: BSPointType
    value: float
    idx: int
    date: str
    zhongshu: object = None
    is_confirmed: bool = True

# ============================================================
#  K线合并处理（包含关系）
# ============================================================

def process_inclusion(df: pd.DataFrame) -> List[ChanKLine]:
    """
    处理K线包含关系，输出合并后的ChanKLine序列
    包含关系：一根K线的高低区间完全包含另一根
    上升趋势：取高高高低（max(high1,high2), max(low1,low2)）
    下降趋势：取低高低低（min(high1,high2), min(low1,low2)）
    """
    if len(df) == 0:
        return []

    chan_klines = []
    for i, row in df.iterrows():
        kl = ChanKLine(
            idx=i,
            date=str(row["Date"])[:10] if pd.notna(row["Date"]) else str(i),
            high=float(row["High"]),
            low=float(row["Low"]),
            open=float(row["Open"]),
            close=float(row["Close"]),
        )
        if len(chan_klines) == 0:
            chan_klines.append(kl)
            continue

        prev = chan_klines[-1]

        # 判断包含关系
        if _is_included(kl, prev):
            # 确定趋势方向
            if len(chan_klines) >= 2:
                pprev = chan_klines[-2]
                trend_up = prev.high >= pprev.high
            else:
                trend_up = kl.high >= prev.high

            merged = _merge_klines(prev, kl, trend_up)
            merged.is_merged = True
            merged.merged_from = getattr(prev, "merged_from", []) + [kl.idx]
            chan_klines[-1] = merged
        else:
            chan_klines.append(kl)

    return chan_klines


def _is_included(a: ChanKLine, b: ChanKLine) -> bool:
    """判断两根K线是否存在包含关系"""
    return (a.high >= b.high and a.low <= b.low) or \
           (b.high >= a.high and b.low <= a.low)


def _merge_klines(a: ChanKLine, b: ChanKLine, trend_up: bool) -> ChanKLine:
    """根据趋势方向合并两根K线"""
    if trend_up:
        return ChanKLine(
            idx=a.idx, date=a.date,
            high=max(a.high, b.high),
            low=max(a.low, b.low),
            open=a.open, close=a.close,
        )
    else:
        return ChanKLine(
            idx=a.idx, date=a.date,
            high=min(a.high, b.high),
            low=min(a.low, b.low),
            open=a.open, close=a.close,
        )

# ============================================================
#  分型识别
# ============================================================

def identify_fractals(chan_klines: List[ChanKLine]) -> List[Fractal]:
    """从合并K线中识别顶分型和底分型"""
    if len(chan_klines) < 3:
        return []

    raw_fractals = []
    for i in range(1, len(chan_klines) - 1):
        prev, curr, nxt = chan_klines[i - 1], chan_klines[i], chan_klines[i + 1]
        # 顶分型：中间高，两侧低
        if curr.high > prev.high and curr.high > nxt.high:
            raw_fractals.append(Fractal(FractalType.TOP, curr, curr.high, i))
        # 底分型：中间低，两侧高
        elif curr.low < prev.low and curr.low < nxt.low:
            raw_fractals.append(Fractal(FractalType.BOTTOM, curr, curr.low, i))

    # 合并连续同类型分型，保留更极端的
    return _merge_adjacent_fractals(raw_fractals)


def _merge_adjacent_fractals(fractals: List[Fractal]) -> List[Fractal]:
    """合并连续同类型分型，保留更极端的那个"""
    if len(fractals) <= 1:
        return fractals

    merged = [fractals[0]]
    for f in fractals[1:]:
        last = merged[-1]
        if f.type == last.type:
            # 同类型，保留更极端的
            if f.type == FractalType.TOP and f.value > last.value:
                merged[-1] = f
            elif f.type == FractalType.BOTTOM and f.value < last.value:
                merged[-1] = f
        else:
            merged.append(f)
    return merged

# ============================================================
#  笔的识别
# ============================================================

def identify_bi(fractals: List[Fractal], chan_klines: List[ChanKLine],
                strict: bool = True) -> List[Bi]:
    """
    从分型序列中识别笔
    strict=True: 严格笔，顶底分型之间至少间隔1根合并K线
    """
    if len(fractals) < 2:
        return []

    bi_list = []
    current = fractals[0]

    for i in range(1, len(fractals)):
        candidate = fractals[i]

        # 必须顶底交替
        if candidate.type == current.type:
            # 同类型，保留更极端的
            if candidate.type == FractalType.TOP and candidate.value > current.value:
                current = candidate
            elif candidate.type == FractalType.BOTTOM and candidate.value < current.value:
                current = candidate
            continue

        # 严格笔检查：间隔至少2个合并K线索引（中间至少1根）
        if strict and candidate.idx - current.idx < 2:
            continue

        # 确保笔的方向与分型一致
        if current.type == FractalType.BOTTOM:
            direction = BiDirection.UP
        else:
            direction = BiDirection.DOWN

        bi = Bi(
            direction=direction,
            start_fractal=current,
            end_fractal=candidate,
            start_idx=chan_klines[current.idx].idx,
            end_idx=chan_klines[candidate.idx].idx,
            start_value=current.value,
            end_value=candidate.value,
            start_date=chan_klines[current.idx].date,
            end_date=chan_klines[candidate.idx].date,
        )
        bi_list.append(bi)
        current = candidate

    return bi_list

# ============================================================
#  线段识别（特征序列法）
# ============================================================

def identify_segments(bi_list: List[Bi]) -> List[Segment]:
    """
    用特征序列法识别线段
    1. 构建特征序列（向上笔取高点，向下笔取低点）
    2. 对特征序列做包含关系处理
    3. 在特征序列中找分型
    4. 分型位置即为线段端点
    """
    if len(bi_list) < 3:
        return []

    # Step 1: 构建特征序列
    eigen_seq = _build_eigen_sequence(bi_list)

    # Step 2: 特征序列包含处理
    merged_eigen = _process_eigen_inclusion(eigen_seq)

    # Step 3: 特征序列分型识别
    eigen_fractals = _identify_eigen_fractals(merged_eigen)

    # Step 4: 根据特征分型确定线段
    segments = []
    if len(eigen_fractals) < 2:
        # 如果特征序列无分型，整段为一个线段
        if len(bi_list) >= 3:
            seg = Segment(
                direction=bi_list[0].direction,
                start_idx=bi_list[0].start_idx,
                end_idx=bi_list[-1].end_idx,
                start_value=bi_list[0].start_value,
                end_value=bi_list[-1].end_value,
                start_date=bi_list[0].start_date,
                end_date=bi_list[-1].end_date,
                bi_count=len(bi_list),
            )
            segments.append(seg)
        return segments

    # 从特征分型映射回笔，构建线段
    seg_start_bi_idx = 0
    for ef in eigen_fractals:
        seg_end_bi_idx = ef["bi_idx"]
        if seg_end_bi_idx <= seg_start_bi_idx:
            continue

        seg_bis = bi_list[seg_start_bi_idx:seg_end_bi_idx + 1]
        if len(seg_bis) >= 1:
            seg = Segment(
                direction=seg_bis[0].direction,
                start_idx=seg_bis[0].start_idx,
                end_idx=seg_bis[-1].end_idx,
                start_value=seg_bis[0].start_value,
                end_value=seg_bis[-1].end_value,
                start_date=seg_bis[0].start_date,
                end_date=seg_bis[-1].end_date,
                bi_count=len(seg_bis),
            )
            segments.append(seg)
        seg_start_bi_idx = seg_end_bi_idx

    # 最后一段
    if seg_start_bi_idx < len(bi_list):
        seg_bis = bi_list[seg_start_bi_idx:]
        if len(seg_bis) >= 1:
            seg = Segment(
                direction=seg_bis[0].direction,
                start_idx=seg_bis[0].start_idx,
                end_idx=seg_bis[-1].end_idx,
                start_value=seg_bis[0].start_value,
                end_value=seg_bis[-1].end_value,
                start_date=seg_bis[0].start_date,
                end_date=seg_bis[-1].end_date,
                bi_count=len(seg_bis),
            )
            segments.append(seg)

    return segments


def _build_eigen_sequence(bi_list: List[Bi]) -> list:
    """构建特征序列：向上笔取high，向下笔取low"""
    eigen = []
    for i, bi in enumerate(bi_list):
        if bi.direction == BiDirection.UP:
            eigen.append({"idx": i, "value": bi.end_value, "high": bi.end_value, "low": bi.start_value})
        else:
            eigen.append({"idx": i, "value": bi.end_value, "high": bi.start_value, "low": bi.end_value})
    return eigen


def _process_eigen_inclusion(eigen_seq: list) -> list:
    """对特征序列做包含关系处理"""
    if len(eigen_seq) <= 1:
        return eigen_seq

    merged = [eigen_seq[0]]
    for i in range(1, len(eigen_seq)):
        curr = eigen_seq[i]
        prev = merged[-1]

        if curr["high"] >= prev["high"] and curr["low"] <= prev["low"]:
            # prev包含curr，保留prev（取更极端的值）
            merged[-1] = {
                "idx": prev["idx"],
                "value": prev["value"],
                "high": max(prev["high"], curr["high"]),
                "low": max(prev["low"], curr["low"]),
            }
        elif prev["high"] >= curr["high"] and prev["low"] <= curr["low"]:
            # curr包含prev
            merged[-1] = {
                "idx": prev["idx"],
                "value": prev["value"],
                "high": min(prev["high"], curr["high"]),
                "low": min(prev["low"], curr["low"]),
            }
        else:
            merged.append(curr)
    return merged


def _identify_eigen_fractals(eigen_seq: list) -> list:
    """在特征序列中识别分型"""
    fractals = []
    for i in range(1, len(eigen_seq) - 1):
        prev, curr, nxt = eigen_seq[i - 1], eigen_seq[i], eigen_seq[i + 1]
        # 顶分型
        if curr["high"] > prev["high"] and curr["high"] > nxt["high"]:
            fractals.append({"type": "top", "bi_idx": curr["idx"], "value": curr["high"]})
        # 底分型
        elif curr["low"] < prev["low"] and curr["low"] < nxt["low"]:
            fractals.append({"type": "bottom", "bi_idx": curr["idx"], "value": curr["low"]})
    return fractals

# ============================================================
#  中枢识别
# ============================================================

def identify_zhongshu(segments: List[Segment]) -> List[ZhongShu]:
    """从线段中识别中枢（至少3个线段的重叠区间）"""
    if len(segments) < 3:
        return []

    zhongshu_list = []
    i = 0

    while i <= len(segments) - 3:
        segs = segments[i:i + 3]

        # 计算三个线段的重叠区间
        highs = [max(s.start_value, s.end_value) for s in segs]
        lows = [min(s.start_value, s.end_value) for s in segs]

        zg = min(highs)  # 中枢上沿
        zd = max(lows)   # 中枢下沿

        if zg <= zd:
            i += 1
            continue  # 无重叠区间

        gg = max(highs)  # 最高点
        dd = min(lows)   # 最低点

        # 尝试延伸中枢
        j = i + 3
        while j < len(segments):
            seg = segments[j]
            seg_high = max(seg.start_value, seg.end_value)
            seg_low = min(seg.start_value, seg.end_value)

            if seg_low <= zg and seg_high >= zd:
                # 仍与中枢有重叠，中枢延伸
                gg = max(gg, seg_high)
                dd = min(dd, seg_low)
                segs.append(seg)
                j += 1
            else:
                break

        zs = ZhongShu(
            zg=round(zg, 4), zd=round(zd, 4),
            gg=round(gg, 4), dd=round(dd, 4),
            start_idx=segs[0].start_idx,
            end_idx=segs[-1].end_idx,
            start_date=segs[0].start_date,
            end_date=segs[-1].end_date,
            segments=segs,
        )
        zhongshu_list.append(zs)
        i = j  # 跳过已构成中枢的线段

    return zhongshu_list

# ============================================================
#  MACD 计算与背驰判断
# ============================================================

def calculate_macd(close: pd.Series, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist


def calculate_macd_area(close: pd.Series, bi: Bi) -> float:
    """计算一笔区间内的MACD面积"""
    dif, dea, hist = calculate_macd(close)
    start, end = bi.start_idx, bi.end_idx
    if start >= len(hist) or end >= len(hist):
        return 0.0
    s, e = min(start, end), max(start, end) + 1
    if s >= e or e > len(hist):
        return 0.0
    if bi.direction == BiDirection.UP:
        area = float(hist.iloc[s:e].clip(lower=0).sum())
    else:
        area = float(abs(hist.iloc[s:e].clip(upper=0).sum()))
    return area


def check_divergence(close: pd.Series, entering_bi: Bi,
                     leaving_bi: Bi, rate: float = 0.9) -> bool:
    """判断背驰：离开笔力度 < 进入笔力度 × rate"""
    entering_area = calculate_macd_area(close, entering_bi)
    leaving_area = calculate_macd_area(close, leaving_bi)
    if entering_area == 0:
        return False
    return (leaving_area / entering_area) < rate

# ============================================================
#  买卖点识别
# ============================================================

def identify_bs_points(bi_list: List[Bi], segments: List[Segment],
                       zhongshu_list: List[ZhongShu],
                       close: pd.Series,
                       divergence_rate: float = 0.9) -> List[BSPoint]:
    """识别缠论买卖点"""
    bs_points = []

    for zs in zhongshu_list:
        # 找进入和离开中枢的笔
        entering_bi = _find_entering_bi(bi_list, zs)
        leaving_bi = _find_leaving_bi(bi_list, zs)

        if entering_bi is None or leaving_bi is None:
            continue

        # 一买/一卖：趋势背驰
        if check_divergence(close, entering_bi, leaving_bi, divergence_rate):
            if leaving_bi.direction == BiDirection.DOWN:
                bs_points.append(BSPoint(
                    BSPointType.BUY_1, leaving_bi.end_value,
                    leaving_bi.end_idx, leaving_bi.end_date, zs
                ))
            else:
                bs_points.append(BSPoint(
                    BSPointType.SELL_1, leaving_bi.end_value,
                    leaving_bi.end_idx, leaving_bi.end_date, zs
                ))

        # 三买/三卖：离开中枢后回调不进中枢
        leaving_bi_after = _find_leaving_bi_after(bi_list, zs)
        if leaving_bi_after is not None:
            callback_low = min(leaving_bi_after.start_value, leaving_bi_after.end_value)
            callback_high = max(leaving_bi_after.start_value, leaving_bi_after.end_value)

            if leaving_bi.direction == BiDirection.UP:
                # 向上离开后回调不进中枢 → 三买
                if callback_low > zs.zg:
                    bs_points.append(BSPoint(
                        BSPointType.BUY_3, callback_low,
                        leaving_bi_after.end_idx, leaving_bi_after.end_date, zs
                    ))
            elif leaving_bi.direction == BiDirection.DOWN:
                # 向下离开后反弹不进中枢 → 三卖
                if callback_high < zs.zd:
                    bs_points.append(BSPoint(
                        BSPointType.SELL_3, callback_high,
                        leaving_bi_after.end_idx, leaving_bi_after.end_date, zs
                    ))

    # 二买/二卖：基于一买一卖之后回调不破
    for i, bsp in enumerate(bs_points[:]):
        if bsp.type == BSPointType.BUY_1:
            # 找一买后第一个回调低点
            subsequent = [b for b in bi_list if b.start_idx > bsp.idx]
            for b in subsequent:
                if b.direction == BiDirection.DOWN:
                    if b.end_value > bsp.value:
                        bs_points.append(BSPoint(
                            BSPointType.BUY_2, b.end_value,
                            b.end_idx, b.end_date, bsp.zhongshu
                        ))
                    break
        elif bsp.type == BSPointType.SELL_1:
            subsequent = [b for b in bi_list if b.start_idx > bsp.idx]
            for b in subsequent:
                if b.direction == BiDirection.UP:
                    if b.end_value < bsp.value:
                        bs_points.append(BSPoint(
                            BSPointType.SELL_2, b.end_value,
                            b.end_idx, b.end_date, bsp.zhongshu
                        ))
                    break

    # 去重排序
    seen = set()
    unique = []
    for bsp in sorted(bs_points, key=lambda x: x.idx):
        key = (bsp.type, bsp.idx)
        if key not in seen:
            seen.add(key)
            unique.append(bsp)
    return unique


def _find_entering_bi(bi_list: List[Bi], zs: ZhongShu) -> Optional[Bi]:
    """找进入中枢的笔（中枢前最后一笔方向与中枢第一段相反）"""
    for bi in reversed(bi_list):
        if bi.end_idx <= zs.start_idx:
            return bi
    return None


def _find_leaving_bi(bi_list: List[Bi], zs: ZhongShu) -> Optional[Bi]:
    """找离开中枢的笔（中枢后第一笔）"""
    for bi in bi_list:
        if bi.start_idx >= zs.end_idx:
            return bi
    return None


def _find_leaving_bi_after(bi_list: List[Bi], zs: ZhongShu) -> Optional[Bi]:
    """找离开中枢后的回调笔"""
    leaving = _find_leaving_bi(bi_list, zs)
    if leaving is None:
        return None
    # 找离开笔之后的第一笔（回调笔）
    for bi in bi_list:
        if bi.start_idx > leaving.end_idx:
            return bi
    return None

# ============================================================
#  数据获取
# ============================================================

@st.cache_data(ttl=3600, show_spinner=True)
def get_a_stock_data(symbol: str, start_date, end_date) -> Optional[pd.DataFrame]:
    """通过akshare获取A股日线数据"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="qfq"
        )
        df = df.rename(columns={
            "日期": "Date", "开盘": "Open", "最高": "High",
            "最低": "Low", "收盘": "Close", "成交量": "Volume"
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        st.error(f"A股数据获取失败: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=True)
def get_hk_us_stock_data(symbol: str, start_date, end_date) -> Optional[pd.DataFrame]:
    """通过yfinance获取港股/美股日线数据"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            return None
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return None

# ============================================================
#  图表构建
# ============================================================

def build_chanlun_charts(df: pd.DataFrame, chan_klines, bi_list,
                        segments, zhongshu_list, bs_points,
                        dif, dea, hist, show_options):
    """构建缠论分析图表"""
    charts = []

    # ---- 主面板: K线 + 缠论要素 ----
    main_series = []

    # 1. K线数据
    candle_data = []
    for _, row in df.iterrows():
        candle_data.append({
            "time": str(row["Date"])[:10],
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
        })

    # 2. 买卖点 markers
    markers = []
    for bsp in bs_points:
        is_buy = "买" in bsp.type.value
        markers.append({
            "time": bsp.date,
            "position": "belowBar" if is_buy else "aboveBar",
            "color": "#4CAF50" if is_buy else "#F44336",
            "shape": "arrowUp" if is_buy else "arrowDown",
            "text": bsp.type.value,
            "size": 2,
        })
    markers.sort(key=lambda x: x["time"])

    candle_options = {
        "upColor": "#26a69a", "downColor": "#ef5350",
        "borderUpColor": "#26a69a", "borderDownColor": "#ef5350",
        "wickUpColor": "#26a69a", "wickDownColor": "#ef5350",
    }

    main_series.append({
        "type": "Candlestick",
        "data": candle_data,
        "options": candle_options,
        "markers": markers if show_options.get("买卖点", True) else [],
    })

    # 3. 笔 (Line series)
    if show_options.get("笔", True):
        for bi in bi_list:
            color = "#2196F3" if bi.direction == BiDirection.UP else "#F44336"
            main_series.append({
                "type": "Line",
                "data": [
                    {"time": bi.start_date, "value": bi.start_value},
                    {"time": bi.end_date, "value": bi.end_value},
                ],
                "options": {
                    "color": color,
                    "lineWidth": 2,
                    "lineStyle": 0,
                    "crosshairMarkerVisible": False,
                    "lastValueVisible": False,
                    "priceLineVisible": False,
                },
            })

    # 4. 线段 (Line series, dashed)
    if show_options.get("线段", True):
        for seg in segments:
            color = "#0D47A1" if seg.direction == BiDirection.UP else "#B71C1C"
            main_series.append({
                "type": "Line",
                "data": [
                    {"time": seg.start_date, "value": seg.start_value},
                    {"time": seg.end_date, "value": seg.end_value},
                ],
                "options": {
                    "color": color,
                    "lineWidth": 3,
                    "lineStyle": 2,  # dashed
                    "crosshairMarkerVisible": False,
                    "lastValueVisible": False,
                    "priceLineVisible": False,
                },
            })

    # 5. 中枢 (Area series)
    if show_options.get("中枢", True):
        for zs in zhongshu_list:
            # 构建中枢覆盖时间段内的Area数据
            area_data = []
            for _, row in df.iterrows():
                d = str(row["Date"])[:10]
                if d >= zs.start_date and d <= zs.end_date:
                    area_data.append({"time": d, "value": zs.zg})

            if len(area_data) >= 2:
                main_series.append({
                    "type": "Area",
                    "data": area_data,
                    "options": {
                        "topColor": "rgba(255, 152, 0, 0.25)",
                        "bottomColor": "rgba(255, 152, 0, 0.05)",
                        "lineColor": "rgba(255, 152, 0, 0.6)",
                        "lineWidth": 1,
                        "crosshairMarkerVisible": False,
                        "lastValueVisible": False,
                        "priceLineVisible": False,
                    },
                })

    chart_options = {
        "height": 500,
        "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"},
        "grid": {"vertLines": {"color": "#2B2B43"}, "horzLines": {"color": "#2B2B43"}},
        "crosshair": {"mode": 0},
        "timeScale": {"timeVisible": False},
    }
    charts.append({"chart": chart_options, "series": main_series})

    # ---- 成交量面板 ----
    volume_data = []
    for _, row in df.iterrows():
        volume_data.append({
            "time": str(row["Date"])[:10],
            "value": float(row["Volume"]),
            "color": "#26a69a" if row["Close"] >= row["Open"] else "#ef5350",
        })

    vol_chart = {
        "height": 100,
        "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"},
        "grid": {"vertLines": {"color": "#2B2B43"}, "horzLines": {"color": "#2B2B43"}},
    }
    charts.append({
        "chart": vol_chart,
        "series": [{"type": "Histogram", "data": volume_data, "options": {"priceFormat": {"type": "volume"}}}]
    })

    # ---- MACD面板 ----
    if show_options.get("MACD", True):
        macd_data_dif = []
        macd_data_dea = []
        macd_data_hist = []
        for i in range(len(dif)):
            if pd.isna(dif.iloc[i]):
                continue
            d = str(df.iloc[i]["Date"])[:10]
            macd_data_dif.append({"time": d, "value": round(float(dif.iloc[i]), 4)})
            macd_data_dea.append({"time": d, "value": round(float(dea.iloc[i]), 4)})
            h = float(hist.iloc[i])
            macd_data_hist.append({
                "time": d,
                "value": round(h, 4),
                "color": "#26a69a" if h >= 0 else "#ef5350",
            })

        macd_chart = {
            "height": 150,
            "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"},
            "grid": {"vertLines": {"color": "#2B2B43"}, "horzLines": {"color": "#2B2B43"}},
        }
        charts.append({
            "chart": macd_chart,
            "series": [
                {"type": "Line", "data": macd_data_dif, "options": {"color": "#2196F3", "lineWidth": 1, "title": "DIF"}},
                {"type": "Line", "data": macd_data_dea, "options": {"color": "#FF9800", "lineWidth": 1, "title": "DEA"}},
                {"type": "Histogram", "data": macd_data_hist, "options": {"title": "MACD"}},
            ]
        })

    return charts

# ============================================================
#  Streamlit 页面
# ============================================================

st.set_page_config(page_title="缠论自动分析", page_icon="🔮", layout="wide")

st.title("🔮 缠论自动分析")
st.markdown("基于缠中说禅理论，自动识别笔、线段、中枢，判断背驰与买卖点")

# ---- 侧边栏 ----
st.sidebar.header("📊 数据源")
market = st.sidebar.selectbox("市场类型", ["A股", "港股", "美股"], index=0)

if market == "A股":
    default_code = "000001"
    code_hint = "6位数字，如 000001、600519"
elif market == "港股":
    default_code = "0700.HK"
    code_hint = "如 0700.HK、9988.HK"
else:
    default_code = "AAPL"
    code_hint = "如 AAPL、MSFT"

symbol = st.sidebar.text_input("股票代码", value=default_code, help=code_hint)
start_date = st.sidebar.date_input("开始日期", value=datetime(2023, 1, 1))
end_date = st.sidebar.date_input("结束日期", value=datetime.today())

st.sidebar.header("⚙️ 缠论参数")
strict_bi = st.sidebar.selectbox("笔识别模式", ["严格笔", "简单笔"], index=0) == "严格笔"
divergence_rate = st.sidebar.slider("背驰比率阈值", 0.5, 1.0, 0.9, 0.05,
                                     help="离开笔力度/进入笔力度 < 阈值则判定背驰")

st.sidebar.header("🎨 显示控制")
show_bi = st.sidebar.checkbox("显示笔", value=True)
show_segment = st.sidebar.checkbox("显示线段", value=True)
show_zhongshu = st.sidebar.checkbox("显示中枢", value=True)
show_bs = st.sidebar.checkbox("显示买卖点", value=True)
show_macd = st.sidebar.checkbox("显示MACD", value=True)

run_button = st.sidebar.button("🚀 开始分析", type="primary", use_container_width=True)

show_options = {
    "笔": show_bi,
    "线段": show_segment,
    "中枢": show_zhongshu,
    "买卖点": show_bs,
    "MACD": show_macd,
}

# ---- 主逻辑 ----
if run_button:
    # 1. 获取数据
    with st.spinner("正在获取数据..."):
        if market == "A股":
            df = get_a_stock_data(symbol, str(start_date), str(end_date))
        else:
            df = get_hk_us_stock_data(symbol, str(start_date), str(end_date))

    if df is None or len(df) < 30:
        st.error("数据获取失败或数据量不足（建议至少200根K线）。请检查股票代码和日期范围。")
        st.stop()

    st.success(f"获取到 {len(df)} 根K线数据")

    # 2. K线合并处理
    with st.spinner("正在处理K线包含关系..."):
        chan_klines = process_inclusion(df)

    # 3. 分型识别
    fractals = identify_fractals(chan_klines)

    # 4. 笔识别
    bi_list = identify_bi(fractals, chan_klines, strict=strict_bi)

    # 5. 线段识别
    segments = identify_segments(bi_list)

    # 6. 中枢识别
    zhongshu_list = identify_zhongshu(segments)

    # 7. MACD计算
    close = df["Close"]
    dif, dea, hist = calculate_macd(close)

    # 8. 买卖点识别
    bs_points = identify_bs_points(bi_list, segments, zhongshu_list, close, divergence_rate)

    # ---- 指标卡片 ----
    col1, col2, col3, col4 = st.columns(4)

    current_price = float(df.iloc[-1]["Close"])
    latest_bi_dir = f"{'↑' if bi_list[-1].direction == BiDirection.UP else '↓'} {bi_list[-1].direction.value}" if bi_list else "无"
    latest_zs = f"[{zhongshu_list[-1].zd:.2f}, {zhongshu_list[-1].zg:.2f}]" if zhongshu_list else "无"
    latest_bs = f"{'🟢' if '买' in bs_points[-1].type.value else '🔴'} {bs_points[-1].type.value} @ {bs_points[-1].value:.2f}" if bs_points else "暂无信号"

    col1.metric("当前价格", f"{current_price:.2f}")
    col2.metric("最新笔方向", latest_bi_dir)
    col3.metric("最新中枢", latest_zs)
    col4.metric("最新信号", latest_bs)

    # ---- 图表 ----
    st.subheader("📈 缠论分析图表")
    charts = build_chanlun_charts(
        df, chan_klines, bi_list, segments,
        zhongshu_list, bs_points, dif, dea, hist, show_options
    )
    renderLightweightCharts(charts)

    # ---- 分析结果 ----
    tab1, tab2, tab3 = st.tabs(["🎯 买卖点", "🔄 中枢统计", "📐 笔线段明细"])

    with tab1:
        if bs_points:
            bs_df = pd.DataFrame([{
                "类型": b.type.value,
                "价格": round(b.value, 2),
                "日期": b.date,
                "确认": "✅" if b.is_confirmed else "⏳",
                "关联中枢": f"[{b.zhongshu.zd:.2f}, {b.zhongshu.zg:.2f}]" if b.zhongshu else "",
            } for b in bs_points])
            st.dataframe(bs_df, use_container_width=True, hide_index=True)
        else:
            st.info("未识别到买卖点信号")

    with tab2:
        if zhongshu_list:
            zs_df = pd.DataFrame([{
                "序号": i + 1,
                "上沿ZG": round(zs.zg, 2),
                "下沿ZD": round(zs.zd, 2),
                "最高GG": round(zs.gg, 2),
                "最低DD": round(zs.dd, 2),
                "振幅%": round((zs.zg - zs.zd) / zs.zd * 100, 2),
                "起始日期": zs.start_date,
                "结束日期": zs.end_date,
                "构成线段数": len(zs.segments),
            } for i, zs in enumerate(zhongshu_list)])
            st.dataframe(zs_df, use_container_width=True, hide_index=True)
        else:
            st.info("未识别到中枢结构")

    with tab3:
        bi_seg_df = pd.DataFrame([{
            "类型": "笔",
            "方向": "↑" if b.direction == BiDirection.UP else "↓",
            "起始价格": round(b.start_value, 2),
            "结束价格": round(b.end_value, 2),
            "起始日期": b.start_date,
            "结束日期": b.end_date,
        } for b in bi_list] + [{
            "类型": "线段",
            "方向": "↑" if s.direction == BiDirection.UP else "↓",
            "起始价格": round(s.start_value, 2),
            "结束价格": round(s.end_value, 2),
            "起始日期": s.start_date,
            "结束日期": s.end_date,
        } for s in segments])
        st.dataframe(bi_seg_df, use_container_width=True, hide_index=True)

    # ---- CSV下载 ----
    col_d1, col_d2, col_d3 = st.columns(3)
    if bs_points:
        bs_csv = pd.DataFrame([{"类型": b.type.value, "价格": b.value, "日期": b.date} for b in bs_points]).to_csv(index=False)
        col_d1.download_button("📥 下载买卖点", bs_csv, "chanlun_bs_points.csv", "text/csv")
    if zhongshu_list:
        zs_csv = pd.DataFrame([{"ZG": zs.zg, "ZD": zs.zd, "GG": zs.gg, "DD": zs.dd, "起始": zs.start_date, "结束": zs.end_date} for zs in zhongshu_list]).to_csv(index=False)
        col_d2.download_button("📥 下载中枢数据", zs_csv, "chanlun_zhongshu.csv", "text/csv")
    bi_csv = pd.DataFrame([{"方向": b.direction.value, "起始": b.start_value, "结束": b.end_value, "起始日期": b.start_date, "结束日期": b.end_date} for b in bi_list]).to_csv(index=False)
    col_d3.download_button("📥 下载笔线段数据", bi_csv, "chanlun_bi_segment.csv", "text/csv")

    # ---- 统计摘要 ----
    st.markdown("---")
    st.subheader("📊 分析摘要")
    summary_col1, summary_col2, summary_col3, summary_col4, summary_col5 = st.columns(5)
    summary_col1.metric("原始K线数", len(df))
    summary_col2.metric("合并后K线数", len(chan_klines))
    summary_col3.metric("识别笔数", len(bi_list))
    summary_col4.metric("识别线段数", len(segments))
    summary_col5.metric("识别中枢数", len(zhongshu_list))

else:
    st.info("👈 在左侧设置参数后，点击 **🚀 开始分析** 进行缠论自动分析")

    st.markdown("""
    ### 🔮 功能说明

    本页面实现 **缠中说禅理论** 的自动分析，包括：

    | 功能 | 说明 |
    |------|------|
    | **笔识别** | 处理K线包含关系 → 识别顶底分型 → 连接形成笔 |
    | **线段识别** | 特征序列法判断线段破坏 |
    | **中枢识别** | 至少3个线段的重叠区间，支持中枢延伸判断 |
    | **背驰判断** | MACD面积法比较进入/离开中枢的力度 |
    | **买卖点** | 一买/一卖(趋势背驰)、二买/二卖(回调不破)、三买/三卖(离开中枢回调不进) |

    > 💡 建议使用 **200根以上K线** 以获得更准确的分析结果
    """)

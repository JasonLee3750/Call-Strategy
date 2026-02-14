import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.stats import norm

TRADING_DAYS = 252


@dataclass
class StrategyParams:
    target_delta: float
    dte: int
    vol_rank_min: float
    vol_rank_max: float
    trend_min: float
    trend_max: float
    take_profit_pct: float = 0.70
    roll_delta: float = 0.30
    roll_spot_ratio: float = 0.98
    hard_defense_dte: int = 5
    hard_defense_spot_ratio: float = 0.99
    dte_tolerance: int = 3


@dataclass
class StrategySummary:
    params: StrategyParams
    trades: int
    cagr: float
    annual_return: float
    assignment_rate: float
    win_rate: float
    max_drawdown: float
    premium_yield_avg: float
    score: float


@dataclass
class LiveOptionCandidate:
    expiration: str
    dte: int
    strike: float
    premium: float
    iv: float
    delta: float
    assignment_prob: float
    annualized_yield: float
    volume: float
    open_interest: float
    score: float


@dataclass
class PositionAdvice:
    expiration: str
    dte: int
    strike: float
    matched_strike: float
    contracts: int
    premium_collected: float
    mark: float
    delta: float
    iv: float
    spot: float
    pnl_cash: float
    pnl_pct: float
    close_trigger_mark: float
    action: str
    reason: str


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def parse_date(v) -> Optional[date]:
    try:
        ts = pd.to_datetime(v, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


def bs_call_metrics(
    spot: float,
    strike: float,
    t_years: float,
    sigma: float,
    r: float = 0.03,
) -> Tuple[float, float, float]:
    """Return (price, delta, ITM probability at expiry)."""
    if spot <= 0 or strike <= 0 or t_years <= 0:
        return 0.0, 1.0, 1.0

    sigma = max(float(sigma), 1e-4)
    root_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t_years) / (sigma * root_t)
    d2 = d1 - sigma * root_t

    call = spot * norm.cdf(d1) - strike * math.exp(-r * t_years) * norm.cdf(d2)
    delta = norm.cdf(d1)
    itm_prob = norm.cdf(d2)

    return max(float(call), 0.0), clamp(float(delta), 0.0, 1.0), clamp(float(itm_prob), 0.0, 1.0)


def trend_label(score: float) -> str:
    if score >= 0.60:
        return "强上行"
    if score >= 0.20:
        return "温和上行"
    if score <= -0.60:
        return "强下行"
    if score <= -0.20:
        return "温和下行"
    return "震荡"


def dynamic_delta(target_delta: float, trend_score: float) -> float:
    # 趋势越强（尤其上行），delta越保守（更远OTM）
    return clamp(target_delta - 0.05 * trend_score, 0.05, 0.35)


def suggested_contracts(total_shares: int, vol_rank_now: float) -> int:
    max_contracts = max(total_shares // 100, 0)
    if max_contracts == 0:
        return 0
    cover_ratio = 0.50 if vol_rank_now < 0.10 else 0.75
    return max(1, min(max_contracts, int(math.floor(max_contracts * cover_ratio + 1e-9))))


def get_next_earnings_date(ticker: str) -> Optional[date]:
    tk = yf.Ticker(ticker)
    today = datetime.utcnow().date()
    dates: List[date] = []

    try:
        edf = tk.get_earnings_dates(limit=12)
        if isinstance(edf, pd.DataFrame) and not edf.empty:
            for d in pd.to_datetime(edf.index, errors="coerce"):
                if pd.isna(d):
                    continue
                dd = d.date()
                if dd >= today:
                    dates.append(dd)
    except Exception:
        pass

    try:
        cal = tk.calendar
        if isinstance(cal, pd.DataFrame):
            values = cal.values.flatten().tolist()
        elif isinstance(cal, dict):
            values = list(cal.values())
        else:
            values = []

        for raw in values:
            items = raw if isinstance(raw, (list, tuple, np.ndarray, pd.Series)) else [raw]
            for item in items:
                dd = parse_date(item)
                if dd and dd >= today:
                    dates.append(dd)
    except Exception:
        pass

    if not dates:
        return None
    return min(dates)


def load_close_history(ticker: str, years: int = 8) -> pd.Series:
    start = (datetime.utcnow() - timedelta(days=years * 365 + 450)).strftime("%Y-%m-%d")
    hist = yf.Ticker(ticker).history(start=start, auto_adjust=True)
    if hist.empty or "Close" not in hist.columns:
        raise ValueError("无法获取历史价格数据。")
    close = hist["Close"].dropna()
    if len(close) < 1000:
        raise ValueError("历史数据长度不足，无法稳定回测8年。")
    return close


def build_feature_frame(close: pd.Series) -> pd.DataFrame:
    ret = close.pct_change()
    hv20 = ret.rolling(20).std() * math.sqrt(TRADING_DAYS)
    hv60 = ret.rolling(60).std() * math.sqrt(TRADING_DAYS)

    vol_rank = pd.Series(index=close.index, dtype=float)
    for i in range(252, len(close)):
        window = hv20.iloc[i - 252 : i].dropna()
        if window.empty or pd.isna(hv20.iloc[i]):
            vol_rank.iloc[i] = np.nan
        else:
            vol_rank.iloc[i] = float((window <= hv20.iloc[i]).mean())

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    trend = (ema20 / ema60) - 1.0
    mom63 = close.pct_change(63)

    max120 = close.rolling(120).max()
    min120 = close.rolling(120).min()
    range_pos120 = (close - min120) / (max120 - min120 + 1e-9)

    trend_score = (
        0.45 * (trend / 0.10).clip(-1.5, 1.5)
        + 0.35 * (mom63 / 0.20).clip(-1.5, 1.5)
        + 0.20 * ((range_pos120 * 2.0 - 1.0).clip(-1.0, 1.0))
    ).clip(-1.5, 1.5)

    feat = pd.DataFrame(
        {
            "close": close,
            "hv20": hv20,
            "hv60": hv60,
            "vol_rank": vol_rank,
            "trend_score": trend_score,
            "trend": trend,
            "mom63": mom63,
            "range_pos120": range_pos120,
        }
    )
    return feat.dropna()


def next_trade_date(index: pd.DatetimeIndex, start_dt: pd.Timestamp, dte: int) -> Optional[pd.Timestamp]:
    target = start_dt + pd.Timedelta(days=dte)
    pos = index.searchsorted(target)
    if pos >= len(index):
        return None
    return index[pos]


def strike_from_delta(spot: float, delta_target: float, sigma: float, t_years: float, r: float = 0.03) -> float:
    z = norm.ppf(clamp(delta_target, 0.02, 0.98))
    ln_s_k = z * sigma * math.sqrt(t_years) - (r + 0.5 * sigma * sigma) * t_years
    return float(spot / math.exp(ln_s_k))


def backtest_single(feat: pd.DataFrame, params: StrategyParams, shares: int) -> Tuple[pd.DataFrame, StrategySummary]:
    close = feat["close"]
    idx = close.index

    start_date = idx.min() + pd.Timedelta(days=365)
    end_date = idx.max() - pd.Timedelta(days=params.dte + 3)
    rebalance_dates = close.loc[start_date:end_date].resample("MS").first().dropna().index

    rows: List[Dict[str, object]] = []

    for dt in rebalance_dates:
        if dt not in feat.index:
            continue

        fr = feat.loc[dt]
        vol_rank = float(fr["vol_rank"])
        trend_score = float(fr["trend_score"])

        if not (params.vol_rank_min <= vol_rank <= params.vol_rank_max):
            continue
        if not (params.trend_min <= trend_score <= params.trend_max):
            continue

        spot = float(fr["close"])
        sigma = float(max(fr["hv20"], 0.08))
        dte = int(params.dte)
        t_years = dte / 365.0

        effective_delta = dynamic_delta(params.target_delta, trend_score)
        strike = strike_from_delta(spot, effective_delta, sigma, t_years)
        premium, model_delta, itm_prob = bs_call_metrics(spot, strike, t_years, sigma)

        expiry = next_trade_date(idx, dt, dte)
        if expiry is None:
            continue
        spot_expiry = float(close.loc[expiry])

        pnl_per_share = premium - max(spot_expiry - strike, 0.0)
        contracts = suggested_contracts(shares, vol_rank)
        pnl_cash = pnl_per_share * contracts * 100.0
        notional = spot * shares
        ret_notional = pnl_cash / (notional + 1e-9)

        rows.append(
            {
                "trade_date": dt.date(),
                "expiry_date": expiry.date(),
                "spot": spot,
                "spot_expiry": spot_expiry,
                "vol_rank": vol_rank,
                "trend_score": trend_score,
                "trend_label": trend_label(trend_score),
                "target_delta_effective": effective_delta,
                "model_delta": model_delta,
                "strike": strike,
                "premium": premium,
                "assignment_prob_model": itm_prob,
                "assigned": int(spot_expiry > strike),
                "contracts": contracts,
                "premium_cash": premium * contracts * 100.0,
                "pnl_per_share": pnl_per_share,
                "pnl_cash": pnl_cash,
                "return_on_notional": ret_notional,
                "premium_yield_annualized": (premium / spot) * (365.0 / dte),
            }
        )

    bt = pd.DataFrame(rows)
    if bt.empty:
        summary = StrategySummary(
            params=params,
            trades=0,
            cagr=-1.0,
            annual_return=-1.0,
            assignment_rate=1.0,
            win_rate=0.0,
            max_drawdown=-1.0,
            premium_yield_avg=0.0,
            score=1e9,
        )
        return bt, summary

    bt["equity"] = (1.0 + bt["return_on_notional"]).cumprod()
    bt["cum_pnl_cash"] = bt["pnl_cash"].cumsum()

    drawdown = bt["equity"] / bt["equity"].cummax() - 1.0
    years_real = max(
        (pd.to_datetime(bt["expiry_date"]).iloc[-1] - pd.to_datetime(bt["trade_date"]).iloc[0]).days / 365.25,
        1.0,
    )

    total = float(bt["equity"].iloc[-1])
    cagr = total ** (1.0 / years_real) - 1.0
    annual_return = float(bt["return_on_notional"].mean() * (len(bt) / years_real))
    assignment_rate = float(bt["assigned"].mean())
    win_rate = float((bt["pnl_cash"] > 0).mean())
    max_dd = float(drawdown.min())
    premium_yield_avg = float(bt["premium_yield_annualized"].mean())

    # 目标：稳定 + 低指派 + 可持续收益
    score = (
        max(0.0, 0.05 - cagr) * 8.0
        + assignment_rate * 2.4
        + abs(max_dd) * 1.6
        + max(0.0, 0.01 - annual_return) * 4.0
        - min(cagr, 0.35) * 1.2
        - min(premium_yield_avg, 0.30) * 0.25
    )

    summary = StrategySummary(
        params=params,
        trades=int(len(bt)),
        cagr=float(cagr),
        annual_return=float(annual_return),
        assignment_rate=assignment_rate,
        win_rate=win_rate,
        max_drawdown=max_dd,
        premium_yield_avg=premium_yield_avg,
        score=float(score),
    )
    return bt, summary


def optimize_strategy(feat: pd.DataFrame, shares: int) -> Tuple[StrategySummary, pd.DataFrame, List[StrategySummary]]:
    deltas = [0.08, 0.10, 0.12, 0.15, 0.18, 0.22]
    dtes = [10, 14, 21, 28, 35]
    vol_windows = [(0.00, 1.00), (0.00, 0.85), (0.20, 0.95), (0.30, 1.00)]
    trend_windows = [(-1.50, 1.50), (-0.40, 1.20), (-0.20, 1.50), (0.00, 1.50)]

    all_res: List[StrategySummary] = []
    bt_map: Dict[Tuple[float, int, float, float, float, float], pd.DataFrame] = {}

    for d in deltas:
        for dte in dtes:
            for vmin, vmax in vol_windows:
                for tmin, tmax in trend_windows:
                    params = StrategyParams(
                        target_delta=d,
                        dte=dte,
                        vol_rank_min=vmin,
                        vol_rank_max=vmax,
                        trend_min=tmin,
                        trend_max=tmax,
                    )
                    bt, summary = backtest_single(feat, params, shares)
                    key = (d, dte, vmin, vmax, tmin, tmax)
                    bt_map[key] = bt
                    all_res.append(summary)

    valid = [
        r
        for r in all_res
        if r.trades >= 30 and r.cagr >= 0.05 and r.assignment_rate <= 0.35 and r.max_drawdown >= -0.35
    ]
    candidates = valid if valid else all_res
    candidates = sorted(candidates, key=lambda x: x.score)
    best = candidates[0]

    key = (
        best.params.target_delta,
        best.params.dte,
        best.params.vol_rank_min,
        best.params.vol_rank_max,
        best.params.trend_min,
        best.params.trend_max,
    )
    return best, bt_map[key], sorted(all_res, key=lambda x: x.score)[:12]


def fetch_live_option_candidates(
    ticker: str,
    params: StrategyParams,
    min_annualized_yield: float,
    max_assignment_prob: float,
) -> Tuple[LiveOptionCandidate, pd.DataFrame, float, Dict[str, float]]:
    tk = yf.Ticker(ticker)
    px = tk.history(period="2y", auto_adjust=True)
    if px.empty:
        raise ValueError("无法获取最新价格。")

    close = px["Close"].dropna()
    spot = float(close.iloc[-1])
    feat_now = build_feature_frame(close)
    if feat_now.empty:
        raise ValueError("历史价格不足，无法计算当前状态。")

    trend_score = float(feat_now["trend_score"].iloc[-1])
    vol_rank = float(feat_now["vol_rank"].iloc[-1])
    hv20 = float(feat_now["hv20"].iloc[-1])
    target_delta_live = dynamic_delta(params.target_delta, trend_score)
    trend_in_window = 1.0 if params.trend_min <= trend_score <= params.trend_max else 0.0
    effective_dte_min = max(7, params.dte - params.dte_tolerance)
    effective_dte_max = min(70, params.dte + params.dte_tolerance)

    expirations = tk.options
    if not expirations:
        raise ValueError("无法获取期权到期日。")

    today = datetime.utcnow().date()
    candidates: List[LiveOptionCandidate] = []

    for exp in expirations:
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte < effective_dte_min or dte > effective_dte_max:
            continue

        calls = tk.option_chain(exp).calls.copy()
        if calls.empty:
            continue

        cols = ["strike", "bid", "ask", "lastPrice", "impliedVolatility", "volume", "openInterest", "delta"]
        for c in cols:
            if c not in calls.columns:
                calls[c] = np.nan

        for _, row in calls.iterrows():
            strike = safe_float(row.get("strike"), np.nan)
            bid = safe_float(row.get("bid"), np.nan)
            ask = safe_float(row.get("ask"), np.nan)
            iv = safe_float(row.get("impliedVolatility"), np.nan)
            delta_raw = safe_float(row.get("delta"), np.nan)
            volume = safe_float(row.get("volume"), 0.0)
            oi = safe_float(row.get("openInterest"), 0.0)

            if np.isnan(strike) or strike <= spot:
                continue
            if np.isnan(bid) or np.isnan(ask) or bid <= 0 or ask <= 0 or ask < bid:
                continue
            premium = 0.5 * (bid + ask)

            sigma = iv if not np.isnan(iv) and iv > 0 else max(hv20, 0.20)
            _, model_delta, itm_prob = bs_call_metrics(spot, strike, dte / 365.0, sigma)
            delta = abs(delta_raw) if not np.isnan(delta_raw) else model_delta
            assignment_prob = max(itm_prob, min(delta, 1.0))
            annualized_yield = (premium / spot) * (365.0 / max(dte, 1))

            if annualized_yield < min_annualized_yield:
                continue
            if assignment_prob > max_assignment_prob:
                continue

            score = (
                abs(delta - target_delta_live) * 1.8
                + abs(dte - params.dte) / 30.0 * 0.8
                + assignment_prob * 0.25
                - min(annualized_yield, 0.60) * 0.2
                + (0.05 if (volume <= 0 and oi <= 0) else 0.0)
            )

            candidates.append(
                LiveOptionCandidate(
                    expiration=exp,
                    dte=dte,
                    strike=float(strike),
                    premium=float(premium),
                    iv=float(sigma),
                    delta=float(delta),
                    assignment_prob=float(assignment_prob),
                    annualized_yield=float(annualized_yield),
                    volume=float(volume),
                    open_interest=float(oi),
                    score=float(score),
                )
            )

    if not candidates:
        raise ValueError(
            f"当前没有满足条件的可卖Call（DTE严格窗口: {effective_dte_min}-{effective_dte_max}天）。"
            " 可放宽年化/行权概率上限，或等待更合适到期日。"
        )

    candidates.sort(key=lambda x: x.score)
    best = candidates[0]

    top_df = pd.DataFrame(
        [
            {
                "expiration": c.expiration,
                "dte": c.dte,
                "strike": c.strike,
                "premium": c.premium,
                "delta": c.delta,
                "annualized_yield": c.annualized_yield,
                "assignment_prob": c.assignment_prob,
                "iv": c.iv,
                "volume": c.volume,
                "open_interest": c.open_interest,
            }
            for c in candidates[:12]
        ]
    )

    ctx = {
        "trend_score": trend_score,
        "vol_rank": vol_rank,
        "hv20": hv20,
        "target_delta_live": target_delta_live,
        "dte_window_min": float(effective_dte_min),
        "dte_window_max": float(effective_dte_max),
        "trend_in_window": trend_in_window,
    }
    return best, top_df, spot, ctx


def build_step_by_step_plan(
    ticker: str,
    shares: int,
    spot: float,
    params: StrategyParams,
    recommendation: LiveOptionCandidate,
    market_ctx: Dict[str, float],
    next_earnings: Optional[date],
) -> List[str]:
    today = datetime.utcnow().date()
    contracts_max = shares // 100
    contracts = suggested_contracts(shares, float(market_ctx.get("vol_rank", 0.5)))
    uncovered = max(shares - contracts * 100, 0)

    close_trigger_mark = max(recommendation.premium * (1.0 - params.take_profit_pct), 0.01)
    soft_roll_spot = recommendation.strike * params.roll_spot_ratio
    hard_roll_spot = recommendation.strike * params.hard_defense_spot_ratio

    lines: List[str] = []
    lines.append(
        f"仓位分配：共 {shares} 股 `{ticker}`，上限 {contracts_max} 张；当前建议先卖 {contracts} 张，保留 {uncovered} 股不覆盖。"
    )

    if next_earnings is None:
        lines.append("财报规则：未获取到下次财报日期，按常规执行但开仓前请手动确认财报时间。")
    else:
        days = (next_earnings - today).days
        if 0 <= days <= 7:
            lines.append(f"财报规则：下次财报约 {next_earnings}（{days}天后），进入禁卖窗口，暂停新开仓。")
        else:
            lines.append(f"财报规则：下次财报约 {next_earnings}，当前不在7天禁卖窗口，可开仓。")

    lines.append(
        f"开仓：`Sell to Open` {contracts} 张 `{ticker}` {recommendation.expiration} {recommendation.strike:.2f}C，目标权利金约 ${recommendation.premium:.2f}/股。"
    )
    lines.append(
        f"止盈：当期权价格 <= ${close_trigger_mark:.2f}/股（约锁定70%利润）时，`Buy to Close`。"
    )
    lines.append(
        f"滚动防守：若 Delta >= {params.roll_delta:.2f} 或股价 >= ${soft_roll_spot:.2f}（执行价98%），执行 `Roll Up & Out`（+7到14天并抬高执行价，优先净收credit）。"
    )
    lines.append(
        f"到期硬规则：若 DTE <= {params.hard_defense_dte} 且股价 >= ${hard_roll_spot:.2f}（执行价99%），当日平仓或滚动，不留到到期。"
    )
    lines.append("日常检查：每个交易日收盘前检查 DTE、Delta、与执行价距离，触发即执行。")
    return lines


def pick_mark_delta_iv(calls: pd.DataFrame, wanted_strike: float, spot: float, dte: int) -> Tuple[float, float, float, float]:
    if calls.empty:
        intrinsic = max(spot - wanted_strike, 0.0)
        mark = intrinsic + (0.10 if dte > 0 else 0.0)
        delta = 0.95 if spot > wanted_strike else 0.05
        return mark, delta, 0.35, wanted_strike

    df = calls.copy()
    for c in ["strike", "bid", "ask", "lastPrice", "impliedVolatility", "delta"]:
        if c not in df.columns:
            df[c] = np.nan

    df["distance"] = (df["strike"] - wanted_strike).abs()
    row = df.sort_values("distance").iloc[0]

    matched_strike = safe_float(row.get("strike"), wanted_strike)
    bid = safe_float(row.get("bid"), np.nan)
    ask = safe_float(row.get("ask"), np.nan)
    last = safe_float(row.get("lastPrice"), np.nan)
    iv = safe_float(row.get("impliedVolatility"), np.nan)
    delta_raw = safe_float(row.get("delta"), np.nan)

    if not np.isnan(bid) and not np.isnan(ask) and bid > 0 and ask > 0 and ask >= bid:
        mark = (bid + ask) / 2.0
    elif not np.isnan(last) and last > 0:
        mark = last
    else:
        intrinsic = max(spot - matched_strike, 0.0)
        mark = intrinsic + (0.10 if dte > 0 else 0.0)

    sigma = iv if not np.isnan(iv) and iv > 0 else 0.35
    _, model_delta, _ = bs_call_metrics(spot, matched_strike, max(dte, 1) / 365.0, sigma)
    delta = abs(delta_raw) if not np.isnan(delta_raw) else model_delta

    return float(mark), float(delta), float(sigma), float(matched_strike)


def decide_action(
    params: StrategyParams,
    spot: float,
    strike: float,
    dte: int,
    delta: float,
    pnl_pct: float,
    next_earnings: Optional[date],
) -> Tuple[str, str]:
    today = datetime.utcnow().date()
    days_to_earnings = (next_earnings - today).days if next_earnings is not None else None
    crosses_earnings = days_to_earnings is not None and days_to_earnings >= 0 and dte >= days_to_earnings

    if not np.isnan(pnl_pct) and pnl_pct >= params.take_profit_pct:
        return "Close", "达到止盈阈值（>=70%），先锁定利润。"

    if dte <= 0:
        if spot > strike:
            return "已到期/高概率被行权", "到期且价内。"
        return "已到期", "到期且价外。"

    if crosses_earnings and days_to_earnings is not None and days_to_earnings <= 2:
        return "Close 或 Roll", "仓位将跨财报且距离财报<=2天。"

    if dte <= params.hard_defense_dte and spot >= strike * params.hard_defense_spot_ratio:
        return "Roll Up & Out", "临近到期且接近/进入价内，按硬规则防守。"

    if delta >= params.roll_delta or spot >= strike * params.roll_spot_ratio:
        return "Roll Up & Out", "Delta或价格触发防守阈值。"

    if crosses_earnings and days_to_earnings is not None and days_to_earnings <= 7:
        return "考虑提前Close", "财报前7天窗口，优先控风险。"

    return "Hold", "未触发止盈/风控阈值。"


def evaluate_positions(
    ticker: str,
    params: StrategyParams,
    spot: float,
    positions_df: pd.DataFrame,
    next_earnings: Optional[date],
) -> pd.DataFrame:
    if positions_df.empty:
        return pd.DataFrame()

    tk = yf.Ticker(ticker)
    chain_cache: Dict[str, pd.DataFrame] = {}
    today = datetime.utcnow().date()
    out_rows: List[Dict[str, object]] = []

    for _, row in positions_df.iterrows():
        expiration = str(row.get("expiration", "")).strip()
        strike = safe_float(row.get("strike"), np.nan)
        contracts = int(safe_float(row.get("contracts"), 0))
        premium_collected = safe_float(row.get("premium_collected"), np.nan)

        exp_date = parse_date(expiration)
        if exp_date is None or np.isnan(strike) or np.isnan(premium_collected) or contracts <= 0:
            continue

        dte = (exp_date - today).days
        if expiration not in chain_cache:
            try:
                chain_cache[expiration] = tk.option_chain(expiration).calls.copy()
            except Exception:
                chain_cache[expiration] = pd.DataFrame()

        mark, delta, iv, matched_strike = pick_mark_delta_iv(chain_cache[expiration], strike, spot, dte)

        pnl_per_share = premium_collected - mark
        pnl_cash = pnl_per_share * contracts * 100.0
        pnl_pct = pnl_per_share / premium_collected if premium_collected > 0 else np.nan
        close_trigger_mark = max(premium_collected * (1.0 - params.take_profit_pct), 0.01)

        action, reason = decide_action(
            params=params,
            spot=spot,
            strike=strike,
            dte=dte,
            delta=delta,
            pnl_pct=pnl_pct,
            next_earnings=next_earnings,
        )

        advice = PositionAdvice(
            expiration=expiration,
            dte=dte,
            strike=float(strike),
            matched_strike=float(matched_strike),
            contracts=contracts,
            premium_collected=float(premium_collected),
            mark=float(mark),
            delta=float(delta),
            iv=float(iv),
            spot=float(spot),
            pnl_cash=float(pnl_cash),
            pnl_pct=float(pnl_pct if not np.isnan(pnl_pct) else 0.0),
            close_trigger_mark=float(close_trigger_mark),
            action=action,
            reason=reason,
        )
        out_rows.append(advice.__dict__)

    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows)


def make_strategy_explanation(summary: StrategySummary, target_cagr: float, max_assign: float) -> str:
    if summary.cagr >= target_cagr:
        head = (
            f"结论：回测满足目标（CAGR {pct(summary.cagr)} >= {pct(target_cagr)}），"
            f"行权率 {pct(summary.assignment_rate)} 在上限 {pct(max_assign)} 内。"
        )
    else:
        head = (
            f"结论：回测未达到你设定的目标年化 {pct(target_cagr)}；"
            f"当前最优候选 CAGR 为 {pct(summary.cagr)}。"
        )

    details = [
        f"- Delta `{summary.params.target_delta:.2f}`，DTE `{summary.params.dte}` 天。",
        (
            f"- 波动率过滤 `{summary.params.vol_rank_min:.2f}-{summary.params.vol_rank_max:.2f}`，"
            f"趋势过滤 `{summary.params.trend_min:.2f}-{summary.params.trend_max:.2f}`。"
        ),
        (
            f"- 指标：交易数 `{summary.trades}`，胜率 `{pct(summary.win_rate)}`，"
            f"最大回撤 `{pct(summary.max_drawdown)}`，平均权利金年化 `{pct(summary.premium_yield_avg)}`。"
        ),
        "- 注意：权利金是毛收入，标的大涨导致的价内损失会侵蚀净收益。",
    ]
    return "\n".join([head] + details)


def build_param_mapping_table(
    params: StrategyParams,
    shares: int,
    recommendation: LiveOptionCandidate,
    market_ctx: Dict[str, float],
    next_earnings: Optional[date],
) -> pd.DataFrame:
    contracts = suggested_contracts(shares, float(market_ctx.get("vol_rank", 0.5)))
    close_trigger_mark = max(recommendation.premium * (1.0 - params.take_profit_pct), 0.01)
    soft_roll_spot = recommendation.strike * params.roll_spot_ratio
    hard_roll_spot = recommendation.strike * params.hard_defense_spot_ratio

    earnings_text = str(next_earnings) if next_earnings is not None else "未知"
    dte_min = int(market_ctx.get("dte_window_min", max(7, params.dte - params.dte_tolerance)))
    dte_max = int(market_ctx.get("dte_window_max", min(70, params.dte + params.dte_tolerance)))

    rows = [
        {
            "策略参数": "target_delta + trend_score",
            "当前市场输入": f"{params.target_delta:.2f} + {float(market_ctx.get('trend_score', 0.0)):.2f}",
            "映射结果": f"实时目标Delta = {float(market_ctx.get('target_delta_live', 0.0)):.2f}",
            "对应动作": f"优先匹配Delta接近 {float(market_ctx.get('target_delta_live', 0.0)):.2f} 的Call",
        },
        {
            "策略参数": "dte",
            "当前市场输入": f"{params.dte}天，容差±{params.dte_tolerance}天",
            "映射结果": f"建议到期日 {recommendation.expiration}（{recommendation.dte}天）",
            "对应动作": f"DTE严格窗口 {dte_min}-{dte_max} 天",
        },
        {
            "策略参数": "vol_rank_min/max",
            "当前市场输入": f"{params.vol_rank_min:.2f}-{params.vol_rank_max:.2f} vs 当前 {float(market_ctx.get('vol_rank', 0.0)):.2f}",
            "映射结果": f"覆盖张数建议 {contracts} 张（{shares} 股）",
            "对应动作": "按覆盖比例卖出，避免满仓覆盖",
        },
        {
            "策略参数": "trend_min/max + earnings window",
            "当前市场输入": f"趋势区间 {params.trend_min:.2f}-{params.trend_max:.2f}；财报 {earnings_text}",
            "映射结果": "当前趋势条件满足" if market_ctx.get("trend_in_window", 0.0) >= 0.5 else "当前趋势条件不满足",
            "对应动作": "趋势与财报窗口共同决定是否开仓",
        },
        {
            "策略参数": "take_profit_pct",
            "当前市场输入": f"{params.take_profit_pct:.0%}",
            "映射结果": f"止盈平仓线 mark <= ${close_trigger_mark:.2f}",
            "对应动作": "达到阈值即 Buy to Close",
        },
        {
            "策略参数": "roll_delta + roll_spot_ratio",
            "当前市场输入": f"Delta {params.roll_delta:.2f} / Spot比 {params.roll_spot_ratio:.2f}",
            "映射结果": f"滚动线 Delta>= {params.roll_delta:.2f} 或 Spot>= ${soft_roll_spot:.2f}",
            "对应动作": "触发即 Roll Up & Out",
        },
        {
            "策略参数": "hard_defense_dte + hard_defense_spot_ratio",
            "当前市场输入": f"DTE<= {params.hard_defense_dte} / Spot比 {params.hard_defense_spot_ratio:.2f}",
            "映射结果": f"硬防守线 DTE<= {params.hard_defense_dte} 且 Spot>= ${hard_roll_spot:.2f}",
            "对应动作": "当日平仓或滚动，不留到到期",
        },
    ]
    return pd.DataFrame(rows)


def render_app() -> None:
    st.set_page_config(page_title="TSLA Sell Call 策略程序", layout="wide")
    st.title("TSLA Sell Call 策略程序")
    st.caption("从零重构版：8年回测 + 当前Step-by-Step + 已卖Call Close/Hold/Roll 管理")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ticker = st.text_input("股票代码", value="TSLA").strip().upper()
    with c2:
        shares = int(st.number_input("持股数量", min_value=100, max_value=200000, value=400, step=100))
    with c3:
        target_cagr_pct = st.number_input("目标年化(%)", min_value=1.0, max_value=30.0, value=5.0, step=0.5)
    with c4:
        max_assign_pct = st.number_input("行权概率上限(%)", min_value=1.0, max_value=95.0, value=20.0, step=1.0)

    run = st.button("运行分析", type="primary")
    if run:
        try:
            with st.spinner("加载历史数据并回测..."):
                close = load_close_history(ticker, years=8)
                feat = build_feature_frame(close)
                best_summary, bt, top_summaries = optimize_strategy(feat, shares)

            params = best_summary.params
            target_cagr = target_cagr_pct / 100.0
            max_assign = max_assign_pct / 100.0

            live_best = None
            live_top = pd.DataFrame()
            live_ctx: Dict[str, float] = {}
            live_error = ""
            spot = float(close.iloc[-1])

            with st.spinner("匹配当前市场期权..."):
                try:
                    live_best, live_top, spot, live_ctx = fetch_live_option_candidates(
                        ticker=ticker,
                        params=params,
                        min_annualized_yield=target_cagr,
                        max_assignment_prob=max_assign,
                    )
                except Exception as ex:
                    live_error = str(ex)

            next_earnings = get_next_earnings_date(ticker)
            search_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")

            st.session_state["analysis"] = {
                "id": search_id,
                "ticker": ticker,
                "shares": shares,
                "target_cagr": target_cagr,
                "max_assign": max_assign,
                "best_summary": best_summary,
                "bt": bt,
                "top_summaries": top_summaries,
                "spot": spot,
                "live_best": live_best,
                "live_top": live_top,
                "live_ctx": live_ctx,
                "live_error": live_error,
                "next_earnings": next_earnings,
            }

            if live_best is not None:
                seed_contracts = suggested_contracts(shares, float(live_ctx.get("vol_rank", 0.5)))
                seed_csv = (
                    "expiration,strike,contracts,premium_collected\n"
                    f"{live_best.expiration},{live_best.strike:.2f},{seed_contracts},{live_best.premium:.2f}\n"
                )
            else:
                seed_csv = "expiration,strike,contracts,premium_collected\n"
            st.session_state[f"position_seed_{search_id}"] = seed_csv

        except Exception as e:
            st.error(f"执行失败: {e}")
            return

    if "analysis" not in st.session_state:
        return

    analysis = st.session_state["analysis"]
    best: StrategySummary = analysis["best_summary"]
    bt: pd.DataFrame = analysis["bt"]
    top_summaries: List[StrategySummary] = analysis["top_summaries"]
    live_best: Optional[LiveOptionCandidate] = analysis["live_best"]
    live_top: pd.DataFrame = analysis["live_top"]
    live_ctx: Dict[str, float] = analysis["live_ctx"]
    spot: float = float(analysis["spot"])
    next_earnings: Optional[date] = analysis["next_earnings"]

    st.subheader("策略参数")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Delta", f"{best.params.target_delta:.2f}")
    s2.metric("DTE", f"{best.params.dte} 天")
    s3.metric("波动率过滤", f"{best.params.vol_rank_min:.2f}-{best.params.vol_rank_max:.2f}")
    s4.metric("趋势过滤", f"{best.params.trend_min:.2f}-{best.params.trend_max:.2f}")

    st.subheader("8年回测结果")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CAGR", pct(best.cagr))
    m2.metric("年化均值", pct(best.annual_return))
    m3.metric("行权率", pct(best.assignment_rate))
    m4.metric("最大回撤", pct(best.max_drawdown))

    m5, m6, m7 = st.columns(3)
    m5.metric("交易次数", best.trades)
    m6.metric("胜率", pct(best.win_rate))
    m7.metric("平均权利金年化", pct(best.premium_yield_avg))

    total_pnl = float(bt["pnl_cash"].sum()) if not bt.empty else 0.0
    years_real = max(
        (pd.to_datetime(bt["expiry_date"]).iloc[-1] - pd.to_datetime(bt["trade_date"]).iloc[0]).days / 365.25,
        1.0,
    ) if not bt.empty else 1.0
    avg_annual_pnl = total_pnl / years_real
    avg_notional = float((bt["spot"] * shares).mean()) if not bt.empty else 0.0
    annual_yield = avg_annual_pnl / (avg_notional + 1e-9)

    p1, p2, p3 = st.columns(3)
    p1.metric("累计期权P/L", f"${total_pnl:,.0f}")
    p2.metric("年均期权P/L", f"${avg_annual_pnl:,.0f}")
    p3.metric("期权腿年化收益率", pct(annual_yield))

    st.subheader("结果解读")
    st.markdown(make_strategy_explanation(best, analysis["target_cagr"], analysis["max_assign"]))

    bt_plot = bt.copy()
    if not bt_plot.empty:
        bt_plot["trade_date"] = pd.to_datetime(bt_plot["trade_date"])
        bt_plot["case_id"] = np.arange(1, len(bt_plot) + 1)
        bt_plot["assigned_label"] = bt_plot["assigned"].map({1: "被行权", 0: "未行权"})

        st.subheader("回测净值")
        equity_line = (
            alt.Chart(bt_plot)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("trade_date:T", title="开仓日期"),
                y=alt.Y("equity:Q", title="净值"),
            )
        )
        points = (
            alt.Chart(bt_plot)
            .mark_point(size=70, filled=True)
            .encode(
                x=alt.X("trade_date:T", title="开仓日期"),
                y=alt.Y("equity:Q", title="净值"),
                color=alt.Color("assigned_label:N", title="到期结果"),
                tooltip=[
                    alt.Tooltip("case_id:Q", title="Case"),
                    alt.Tooltip("trade_date:T", title="开仓日"),
                    alt.Tooltip("expiry_date:N", title="到期日"),
                    alt.Tooltip("pnl_cash:Q", format=",.0f", title="P/L($)"),
                    alt.Tooltip("return_on_notional:Q", format=".2%", title="单笔收益"),
                    alt.Tooltip("assigned_label:N", title="是否行权"),
                ],
            )
        )
        st.altair_chart((equity_line + points).interactive(), use_container_width=True)

    st.subheader("策略候选Top12")
    top_df = pd.DataFrame(
        [
            {
                "delta": r.params.target_delta,
                "dte": r.params.dte,
                "vol_rank_min": r.params.vol_rank_min,
                "vol_rank_max": r.params.vol_rank_max,
                "trend_min": r.params.trend_min,
                "trend_max": r.params.trend_max,
                "trades": r.trades,
                "cagr": r.cagr,
                "assignment_rate": r.assignment_rate,
                "max_drawdown": r.max_drawdown,
                "score": r.score,
            }
            for r in top_summaries
        ]
    )
    st.dataframe(
        top_df.style.format(
            {
                "cagr": "{:.2%}",
                "assignment_rate": "{:.2%}",
                "max_drawdown": "{:.2%}",
                "score": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    if live_best is None:
        st.warning(f"当前市场建议生成失败：{analysis['live_error']}")
    else:
        st.subheader("当前市场建议")
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Spot", f"${spot:.2f}")
        l2.metric("趋势分数", f"{float(live_ctx.get('trend_score', 0.0)):.2f}")
        l3.metric("波动率分位", f"{float(live_ctx.get('vol_rank', 0.0)):.2f}")
        l4.metric("实时目标Delta", f"{float(live_ctx.get('target_delta_live', 0.0)):.2f}")
        st.caption(
            f"DTE严格窗口: {int(live_ctx.get('dte_window_min', 0))}-{int(live_ctx.get('dte_window_max', 0))} 天；"
            f"当前推荐偏差: {abs(int(live_best.dte) - int(best.params.dte))} 天。"
        )
        st.subheader("参数 -> 当前建议 映射表")
        mapping_df = build_param_mapping_table(
            params=best.params,
            shares=shares,
            recommendation=live_best,
            market_ctx=live_ctx,
            next_earnings=next_earnings,
        )
        st.dataframe(mapping_df, use_container_width=True)

        rec = {
            "Ticker": analysis["ticker"],
            "Expiration": live_best.expiration,
            "DTE": live_best.dte,
            "Strike": f"${live_best.strike:.2f}",
            "Premium": f"${live_best.premium:.2f}",
            "Delta": f"{live_best.delta:.2f}",
            "估计行权概率": pct(live_best.assignment_prob),
            "权利金年化": pct(live_best.annualized_yield),
            "IV": pct(live_best.iv),
            "下次财报": str(next_earnings) if next_earnings else "未知",
        }
        st.table(pd.DataFrame([rec]))

        st.subheader("Step-by-Step 操作清单")
        steps = build_step_by_step_plan(
            ticker=analysis["ticker"],
            shares=shares,
            spot=spot,
            params=best.params,
            recommendation=live_best,
            market_ctx=live_ctx,
            next_earnings=next_earnings,
        )
        for i, line in enumerate(steps, start=1):
            st.markdown(f"{i}. {line}")

        st.subheader("当前备选Call Top12")
        st.dataframe(
            live_top.style.format(
                {
                    "strike": "${:.2f}",
                    "premium": "${:.2f}",
                    "delta": "{:.2f}",
                    "annualized_yield": "{:.2%}",
                    "assignment_prob": "{:.2%}",
                    "iv": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

        st.subheader("已卖出Call管理：Close / Hold / Roll")
        st.caption("CSV格式：expiration,strike,contracts,premium_collected（premium_collected按每股填写）")

        sid = analysis["id"]
        csv_key = f"positions_csv_{sid}"
        default_csv = st.session_state.get(f"position_seed_{sid}", "expiration,strike,contracts,premium_collected\n")
        csv_text = st.text_area("输入持仓CSV", key=csv_key, value=default_csv, height=130)

        eval_key = f"positions_eval_{sid}"
        if st.button("评估持仓"):
            try:
                pos_df = pd.read_csv(StringIO(csv_text))
                required = {"expiration", "strike", "contracts", "premium_collected"}
                miss = [c for c in required if c not in pos_df.columns]
                if miss:
                    st.error(f"CSV缺少字段: {', '.join(miss)}")
                else:
                    st.session_state[eval_key] = evaluate_positions(
                        ticker=analysis["ticker"],
                        params=best.params,
                        spot=spot,
                        positions_df=pos_df,
                        next_earnings=next_earnings,
                    )
            except Exception as ex:
                st.error(f"持仓评估失败: {ex}")

        if eval_key in st.session_state:
            eval_df = st.session_state[eval_key]
            if eval_df.empty:
                st.info("没有解析到有效持仓，请检查CSV。")
            else:
                st.dataframe(
                    eval_df.style.format(
                        {
                            "strike": "${:.2f}",
                            "matched_strike": "${:.2f}",
                            "premium_collected": "${:.2f}",
                            "mark": "${:.2f}",
                            "delta": "{:.2f}",
                            "iv": "{:.2%}",
                            "spot": "${:.2f}",
                            "pnl_cash": "${:,.0f}",
                            "pnl_pct": "{:.2%}",
                            "close_trigger_mark": "${:.2f}",
                        }
                    ),
                    use_container_width=True,
                )
                st.caption("标准止盈线：mark <= close_trigger_mark。若 Action=Roll Up & Out，按策略先滚动而不是硬扛到期。")

    st.subheader("回测明细")
    st.dataframe(
        bt.style.format(
            {
                "spot": "${:.2f}",
                "spot_expiry": "${:.2f}",
                "vol_rank": "{:.2f}",
                "trend_score": "{:.2f}",
                "target_delta_effective": "{:.2f}",
                "model_delta": "{:.2f}",
                "strike": "${:.2f}",
                "premium": "${:.2f}",
                "assignment_prob_model": "{:.2%}",
                "premium_cash": "${:,.0f}",
                "pnl_per_share": "${:.2f}",
                "pnl_cash": "${:,.0f}",
                "cum_pnl_cash": "${:,.0f}",
                "return_on_notional": "{:.2%}",
                "premium_yield_annualized": "{:.2%}",
                "equity": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    csv = bt.to_csv(index=False).encode("utf-8")
    st.download_button(
        "下载回测CSV",
        data=csv,
        file_name=f"{analysis['ticker']}_sell_call_backtest_8y.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    render_app()

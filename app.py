import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import altair as alt
from scipy.stats import norm

TRADING_DAYS = 252


@dataclass
class StrategyConfig:
    target_delta: float
    dte: int
    vol_rank_min: float
    vol_rank_max: float
    trend_min: float
    trend_max: float


@dataclass
class StrategyResult:
    config: StrategyConfig
    trades: int
    assignment_rate: float
    win_rate: float
    cagr: float
    annualized_mean: float
    max_drawdown: float
    premium_yield_avg: float
    score: float


@dataclass
class LiveOption:
    expiration: str
    dte: int
    strike: float
    premium: float
    iv: float
    assignment_prob: float
    premium_yield_annualized: float
    volume: float
    open_interest: float
    score: float


def clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def safe_float(v, default=np.nan) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def bs_call_price_itm_prob_delta(
    spot: float,
    strike: float,
    t_years: float,
    sigma: float,
    r: float = 0.03,
) -> Tuple[float, float, float]:
    if spot <= 0 or strike <= 0 or t_years <= 0:
        return 0.0, 1.0, 1.0

    sigma = max(sigma, 1e-4)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * t_years) / (sigma * math.sqrt(t_years))
    d2 = d1 - sigma * math.sqrt(t_years)
    call = spot * norm.cdf(d1) - strike * math.exp(-r * t_years) * norm.cdf(d2)
    itm_prob = norm.cdf(d2)
    delta = norm.cdf(d1)
    return max(call, 0.0), float(np.clip(itm_prob, 0.0, 1.0)), float(np.clip(delta, 0.0, 1.0))


def load_price_history(ticker: str, years: int = 8) -> pd.Series:
    start = (datetime.utcnow() - timedelta(days=years * 365 + 450)).strftime("%Y-%m-%d")
    tk = yf.Ticker(ticker)
    hist = tk.history(start=start, auto_adjust=True)
    if hist.empty or len(hist) < 1000:
        raise ValueError("历史数据不足，无法完成8年策略搜索。")
    close = hist["Close"].dropna()
    return close


def compute_features(close: pd.Series) -> pd.DataFrame:
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
    roll_max120 = close.rolling(120).max()
    roll_min120 = close.rolling(120).min()
    range_pos120 = (close - roll_min120) / (roll_max120 - roll_min120 + 1e-9)

    trend_shape = (
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
            "trend": trend,
            "mom63": mom63,
            "range_pos120": range_pos120,
            "trend_shape": trend_shape,
        }
    )
    return feat.dropna()


def trend_label(v: float) -> str:
    if v >= 0.6:
        return "强上行"
    if v >= 0.2:
        return "温和上行"
    if v <= -0.6:
        return "强下行"
    if v <= -0.2:
        return "温和下行"
    return "震荡"


def adjusted_delta(base_delta: float, trend_shape_value: float) -> float:
    # 上行趋势更强时，下调delta（更远OTM）以降低被行权风险。
    return clamp(base_delta - 0.05 * trend_shape_value, 0.05, 0.35)


def find_expiry(index: pd.DatetimeIndex, start_dt: pd.Timestamp, dte: int) -> Optional[pd.Timestamp]:
    target = start_dt + pd.Timedelta(days=dte)
    pos = index.searchsorted(target)
    if pos >= len(index):
        return None
    return index[pos]


def simulate_short_call_leg(
    feat: pd.DataFrame,
    cfg: StrategyConfig,
    years: int = 8,
) -> Tuple[pd.DataFrame, StrategyResult]:
    close = feat["close"]
    idx = close.index
    first = idx.min() + pd.Timedelta(days=365)
    last = idx.max() - pd.Timedelta(days=cfg.dte + 3)

    rebal_dates = close.loc[first:last].resample("MS").first().dropna().index
    rows = []

    for dt in rebal_dates:
        if dt not in feat.index:
            continue
        row = feat.loc[dt]
        vr = float(row["vol_rank"])
        tr = float(row["trend_shape"])
        if vr < cfg.vol_rank_min or vr > cfg.vol_rank_max:
            continue
        if tr < cfg.trend_min or tr > cfg.trend_max:
            continue

        spot = float(row["close"])
        sigma = float(max(row["hv20"], 0.08))
        t_years = cfg.dte / 365.0

        eff_delta = adjusted_delta(cfg.target_delta, tr)
        # 使用动态delta反推执行价（BSM）
        z = norm.ppf(clamp(eff_delta, 0.02, 0.98))
        ln_s_k = z * sigma * math.sqrt(t_years) - (0.03 + 0.5 * sigma**2) * t_years
        strike = spot / math.exp(ln_s_k)

        premium, itm_prob, delta = bs_call_price_itm_prob_delta(spot, strike, t_years, sigma)

        expiry = find_expiry(idx, dt, cfg.dte)
        if expiry is None:
            continue
        st_exp = float(close.loc[expiry])

        pnl = premium - max(st_exp - strike, 0.0)
        ret = pnl / spot
        assigned = 1 if st_exp > strike else 0

        rows.append(
            {
                "trade_date": dt.date(),
                "expiry_date": expiry.date(),
                "spot": spot,
                "vol_rank": vr,
                "trend_shape": tr,
                "trend_label": trend_label(tr),
                "delta_target_effective": eff_delta,
                "strike": strike,
                "premium": premium,
                "delta": delta,
                "assignment_prob_model": itm_prob,
                "spot_at_expiry": st_exp,
                "assigned": assigned,
                "pnl": pnl,
                "return": ret,
                "premium_yield_annualized": (premium / spot) * (365.0 / cfg.dte),
            }
        )

    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt, StrategyResult(cfg, 0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0, -1e9)

    bt["equity"] = (1.0 + bt["return"]).cumprod()
    dd = bt["equity"] / bt["equity"].cummax() - 1.0

    years_real = max((pd.to_datetime(bt["expiry_date"]).iloc[-1] - pd.to_datetime(bt["trade_date"]).iloc[0]).days / 365.25, 1.0)
    total = float(bt["equity"].iloc[-1])
    cagr = total ** (1.0 / years_real) - 1.0
    mean_ret = float(bt["return"].mean())
    trades_per_year = len(bt) / years_real
    annualized_mean = mean_ret * trades_per_year
    assignment_rate = float(bt["assigned"].mean())
    premium_yield_avg = float(bt["premium_yield_annualized"].mean())

    # 评分：必须尽量稳定+低行权，并奖励达到5%年化
    target_gap_penalty = max(0.05 - cagr, 0.0) * 8.0
    score = (
        target_gap_penalty
        + assignment_rate * 2.2
        + abs(float(dd.min())) * 1.4
        + max(0.0, 0.01 - annualized_mean) * 4.0
        - min(cagr, 0.30) * 1.2
        - min(premium_yield_avg, 0.25) * 0.2
    )

    result = StrategyResult(
        config=cfg,
        trades=int(len(bt)),
        assignment_rate=assignment_rate,
        win_rate=float((bt["pnl"] > 0).mean()),
        cagr=float(cagr),
        annualized_mean=float(annualized_mean),
        max_drawdown=float(dd.min()),
        premium_yield_avg=float(premium_yield_avg),
        score=float(score),
    )
    return bt, result


def optimize_strategy(feat: pd.DataFrame) -> Tuple[StrategyResult, pd.DataFrame, List[StrategyResult]]:
    deltas = [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.28]
    dtes = [10, 14, 21, 28, 35, 45]
    vol_windows = [
        (0.00, 1.00),
        (0.00, 0.80),
        (0.20, 0.90),
        (0.30, 0.95),
        (0.40, 1.00),
    ]
    trend_windows = [
        (-1.50, 1.50),
        (-0.40, 1.50),
        (-0.20, 1.50),
        (0.00, 1.50),
        (-1.50, 0.80),
    ]

    all_results: List[StrategyResult] = []
    bt_map: Dict[Tuple[float, int, float, float, float, float], pd.DataFrame] = {}

    for d in deltas:
        for t in dtes:
            for vmin, vmax in vol_windows:
                for tr_min, tr_max in trend_windows:
                    cfg = StrategyConfig(
                        target_delta=d,
                        dte=t,
                        vol_rank_min=vmin,
                        vol_rank_max=vmax,
                        trend_min=tr_min,
                        trend_max=tr_max,
                    )
                    bt, res = simulate_short_call_leg(feat, cfg)
                    all_results.append(res)
                    bt_map[(d, t, vmin, vmax, tr_min, tr_max)] = bt

    valid = [
        r
        for r in all_results
        if r.trades >= 30 and r.cagr >= 0.05 and r.assignment_rate <= 0.35 and r.max_drawdown >= -0.30
    ]

    if valid:
        valid.sort(key=lambda x: x.score)
        best = valid[0]
    else:
        # 如果严格条件下无策略，退化为最优稳健解并提示用户
        all_results.sort(key=lambda x: x.score)
        best = all_results[0]

    k = (
        best.config.target_delta,
        best.config.dte,
        best.config.vol_rank_min,
        best.config.vol_rank_max,
        best.config.trend_min,
        best.config.trend_max,
    )
    return best, bt_map[k], sorted(all_results, key=lambda x: x.score)[:12]


def pick_live_option(
    ticker: str,
    target_delta: float,
    target_dte: int,
    trend_min: float,
    trend_max: float,
    min_premium_annualized: float,
    max_assignment_prob: float,
) -> Tuple[LiveOption, pd.DataFrame, float, Dict[str, float]]:
    tk = yf.Ticker(ticker)
    px = tk.history(period="2y", auto_adjust=True)
    if px.empty:
        raise ValueError("无法获取最新价格。")
    spot = float(px["Close"].iloc[-1])
    feat_now = compute_features(px["Close"].dropna())
    if feat_now.empty:
        raise ValueError("历史数据不足，无法计算当前趋势形态。")
    trend_now = float(feat_now["trend_shape"].iloc[-1])
    trend_name = trend_label(trend_now)
    in_trend_window = trend_min <= trend_now <= trend_max
    target_delta_live = adjusted_delta(target_delta, trend_now)

    exps = tk.options
    if not exps:
        raise ValueError("无法获取期权到期日。")

    now = datetime.utcnow().date()
    candidates: List[LiveOption] = []

    for exp in exps:
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - now).days
        if dte < 7 or dte > 70:
            continue

        chain = tk.option_chain(exp)
        calls = chain.calls.copy()
        if calls.empty:
            continue

        wanted = ["strike", "lastPrice", "impliedVolatility", "volume", "openInterest", "delta"]
        available = [c for c in wanted if c in calls.columns]
        calls = calls[available].copy()
        for c in wanted:
            if c not in calls.columns:
                calls[c] = np.nan

        for _, row in calls.iterrows():
            strike = safe_float(row.get("strike"))
            premium = safe_float(row.get("lastPrice"), 0.0)
            iv = safe_float(row.get("impliedVolatility"), np.nan)
            delta_raw = safe_float(row.get("delta"), np.nan)
            vol = safe_float(row.get("volume"), 0.0)
            oi = safe_float(row.get("openInterest"), 0.0)

            if np.isnan(strike) or strike <= spot or premium <= 0:
                continue

            sigma = iv if not np.isnan(iv) and iv > 0 else 0.35
            t_years = dte / 365.0
            _, itm_prob, bsm_delta = bs_call_price_itm_prob_delta(spot, strike, t_years, sigma)
            delta = abs(delta_raw) if not np.isnan(delta_raw) else bsm_delta

            annualized = (premium / spot) * (365.0 / dte)
            assign_prob = max(itm_prob, min(delta, 1.0))

            if annualized < min_premium_annualized:
                continue
            if assign_prob > max_assignment_prob:
                continue

            score = (
                abs(delta - target_delta_live) * 1.8
                + abs(dte - target_dte) / 30.0 * 0.8
                + assign_prob * 0.25
                - min(annualized, 0.60) * 0.2
                + (0.05 if (vol <= 0 and oi <= 0) else 0.0)
            )

            candidates.append(
                LiveOption(
                    expiration=exp,
                    dte=dte,
                    strike=strike,
                    premium=premium,
                    iv=sigma,
                    assignment_prob=assign_prob,
                    premium_yield_annualized=annualized,
                    volume=vol,
                    open_interest=oi,
                    score=score,
                )
            )

    if not candidates:
        raise ValueError("当前没有满足条件的可卖call。可放宽行权概率或最低年化约束。")

    candidates.sort(key=lambda x: x.score)
    best = candidates[0]

    top = pd.DataFrame(
        [
            {
                "expiration": c.expiration,
                "dte": c.dte,
                "strike": c.strike,
                "premium": c.premium,
                "annualized_yield": c.premium_yield_annualized,
                "assignment_prob": c.assignment_prob,
                "iv": c.iv,
                "volume": c.volume,
                "open_interest": c.open_interest,
            }
            for c in candidates[:12]
        ]
    )
    return best, top, spot, {
        "trend_shape": trend_now,
        "trend_label": trend_name,
        "target_delta_live": target_delta_live,
        "trend_in_window": 1.0 if in_trend_window else 0.0,
    }


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def build_strategy_explanation(
    strategy: StrategyResult,
    user_target_cagr: float,
    user_max_assign: float,
) -> str:
    if strategy.cagr >= user_target_cagr:
        top_line = (
            f"结论：该策略在8年回测中达到你设定的目标（CAGR {pct(strategy.cagr)} >= {pct(user_target_cagr)}），"
            f"且行权率 {pct(strategy.assignment_rate)} 在你设定上限 {pct(user_max_assign)} 以内。"
        )
    else:
        top_line = (
            f"结论：在当前模型与约束下，未达到你设定的目标年化 {pct(user_target_cagr)}；"
            f"当前最稳健候选 CAGR 为 {pct(strategy.cagr)}。"
        )

    delta_text = (
        f"- 目标Delta `{strategy.config.target_delta:.2f}`：数值越低，执行价越远离现价，通常更不易被行权，但权利金也更少。"
    )
    dte_text = f"- 目标DTE `{strategy.config.dte}天`：表示每次卖出约 `{strategy.config.dte}` 天到期的 call。"
    vol_text = (
        f"- 波动率分位过滤 `{strategy.config.vol_rank_min:.2f}-{strategy.config.vol_rank_max:.2f}`："
        "仅在该波动率区间内开仓。"
    )
    trend_text = (
        f"- 价格形态趋势过滤 `{strategy.config.trend_min:.2f}-{strategy.config.trend_max:.2f}`："
        "仅在该趋势形态分数区间开仓；强上行时会自动下调delta，减少被行权概率。"
    )
    perf_text = (
        f"- 收益与风险：8年CAGR `{pct(strategy.cagr)}`，最大回撤 `{pct(strategy.max_drawdown)}`，"
        f"行权率 `{pct(strategy.assignment_rate)}`，胜率 `{pct(strategy.win_rate)}`。"
    )
    premium_gap_text = (
        f"- 为什么“平均权利金年化 `{pct(strategy.premium_yield_avg)}`”可能高于策略CAGR："
        "权利金是毛收入，遇到标的大涨导致的被行权亏损会侵蚀净收益，因此最终复利可能明显低于权利金年化。"
    )
    sample_text = f"- 交易次数 `{strategy.trades}`：样本越多，结论通常越稳健。"

    return "\n".join([top_line, delta_text, dte_text, vol_text, trend_text, perf_text, premium_gap_text, sample_text])


def render_app() -> None:
    st.set_page_config(page_title="8年卖Call策略搜索器", layout="wide")
    st.title("8年历史 + 波动率 + 价格形态 的 Sell Call 策略搜索器")
    st.caption("目标：找到可稳定收权利金、低行权率、且年化>=5%的策略；并给出当前可执行合约（研究用途）。")

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("股票代码", value="TSLA").strip().upper()
    with col2:
        min_target_cagr_pct = st.number_input("最低目标年化(%)", min_value=1.0, max_value=30.0, value=5.0, step=0.5)
    with col3:
        max_assign_pct = st.number_input("行权概率上限(%)", min_value=1.0, max_value=95.0, value=20.0, step=1.0)

    if st.button("搜索策略并给出当前建议", type="primary"):
        try:
            with st.spinner("加载8年历史并搜索最优策略..."):
                close = load_price_history(ticker, years=8)
                feat = compute_features(close)
                best_strategy, bt, top_strategies = optimize_strategy(feat)

            # 动态强制用户目标门槛
            user_target_cagr = min_target_cagr_pct / 100.0
            if best_strategy.cagr < user_target_cagr:
                st.warning(
                    f"在当前模型与约束下，未找到满足你设定年化 {min_target_cagr_pct:.1f}% 的策略。"
                    f" 当前最稳健候选年化约 {best_strategy.cagr*100:.2f}%。"
                )

            st.subheader("筛选出的策略")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("目标Delta", f"{best_strategy.config.target_delta:.2f}")
            s2.metric("目标DTE", f"{best_strategy.config.dte} 天")
            s3.metric("波动率分位过滤", f"{best_strategy.config.vol_rank_min:.2f}-{best_strategy.config.vol_rank_max:.2f}")
            s4.metric("趋势形态过滤", f"{best_strategy.config.trend_min:.2f}-{best_strategy.config.trend_max:.2f}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("8年CAGR", pct(best_strategy.cagr))
            m2.metric("年化均值", pct(best_strategy.annualized_mean))
            m3.metric("行权率", pct(best_strategy.assignment_rate))
            m4.metric("最大回撤", pct(best_strategy.max_drawdown))

            m5, m6, m7 = st.columns(3)
            m5.metric("交易次数", best_strategy.trades)
            m6.metric("胜率", pct(best_strategy.win_rate))
            m7.metric("平均权利金年化", pct(best_strategy.premium_yield_avg))

            st.subheader("结果解读")
            st.markdown(
                build_strategy_explanation(
                    strategy=best_strategy,
                    user_target_cagr=user_target_cagr,
                    user_max_assign=max_assign_pct / 100.0,
                )
            )

            st.subheader("回测净值（含每笔交易Marker）")
            bt_plot = bt.copy()
            bt_plot["trade_date"] = pd.to_datetime(bt_plot["trade_date"])
            bt_plot["case_id"] = np.arange(1, len(bt_plot) + 1)
            bt_plot["assigned_label"] = bt_plot["assigned"].map({1: "被行权", 0: "未行权"})

            equity_line = (
                alt.Chart(bt_plot)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X("trade_date:T", title="开仓日期"),
                    y=alt.Y("equity:Q", title="净值"),
                )
            )
            equity_points = (
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
                        alt.Tooltip("equity:Q", format=".4f", title="净值"),
                        alt.Tooltip("return:Q", format=".2%", title="单笔收益"),
                        alt.Tooltip("assigned_label:N", title="是否行权"),
                    ],
                )
            )
            st.altair_chart((equity_line + equity_points).interactive(), use_container_width=True)

            st.subheader("每个Case收益点图")
            returns_points = (
                alt.Chart(bt_plot)
                .mark_circle(size=90)
                .encode(
                    x=alt.X("trade_date:T", title="开仓日期"),
                    y=alt.Y("return:Q", title="单笔收益率", axis=alt.Axis(format="%")),
                    color=alt.Color("assigned_label:N", title="到期结果"),
                    tooltip=[
                        alt.Tooltip("case_id:Q", title="Case"),
                        alt.Tooltip("trade_date:T", title="开仓日"),
                        alt.Tooltip("expiry_date:N", title="到期日"),
                        alt.Tooltip("strike:Q", format=".2f", title="执行价"),
                        alt.Tooltip("premium:Q", format=".2f", title="权利金"),
                        alt.Tooltip("return:Q", format=".2%", title="单笔收益"),
                        alt.Tooltip("assigned_label:N", title="是否行权"),
                    ],
                )
            )
            zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[6, 4]).encode(y="y:Q")
            st.altair_chart((zero_line + returns_points).interactive(), use_container_width=True)

            st.subheader("策略候选Top12")
            tdf = pd.DataFrame(
                [
                    {
                        "target_delta": r.config.target_delta,
                        "dte": r.config.dte,
                        "vol_rank_min": r.config.vol_rank_min,
                        "vol_rank_max": r.config.vol_rank_max,
                        "trend_min": r.config.trend_min,
                        "trend_max": r.config.trend_max,
                        "trades": r.trades,
                        "cagr": r.cagr,
                        "assignment_rate": r.assignment_rate,
                        "max_drawdown": r.max_drawdown,
                        "score": r.score,
                    }
                    for r in top_strategies
                ]
            )
            st.dataframe(
                tdf.style.format(
                    {
                        "cagr": "{:.2%}",
                        "assignment_rate": "{:.2%}",
                        "max_drawdown": "{:.2%}",
                        "score": "{:.3f}",
                    }
                ),
                use_container_width=True,
            )

            with st.spinner("根据最佳策略匹配当前可卖call..."):
                live_best, live_top, spot, live_ctx = pick_live_option(
                    ticker=ticker,
                    target_delta=best_strategy.config.target_delta,
                    target_dte=best_strategy.config.dte,
                    trend_min=best_strategy.config.trend_min,
                    trend_max=best_strategy.config.trend_max,
                    min_premium_annualized=user_target_cagr,
                    max_assignment_prob=max_assign_pct / 100.0,
                )

            st.subheader("当前形态趋势状态")
            l1, l2, l3 = st.columns(3)
            l1.metric("趋势形态分数", f"{live_ctx['trend_shape']:.2f}")
            l2.metric("形态标签", str(live_ctx["trend_label"]))
            l3.metric("实时目标Delta(调整后)", f"{live_ctx['target_delta_live']:.2f}")
            if live_ctx["trend_in_window"] < 0.5:
                st.info("当前形态趋势不在策略最优开仓区间内，建议等待更匹配的价格形态。")

            st.subheader("当前时刻建议卖出的Call")
            rec = {
                "Ticker": ticker,
                "Spot": f"${spot:.2f}",
                "Expiration": live_best.expiration,
                "DTE": live_best.dte,
                "Strike": f"${live_best.strike:.2f}",
                "Premium": f"${live_best.premium:.2f}",
                "权利金年化": pct(live_best.premium_yield_annualized),
                "估计行权概率": pct(live_best.assignment_prob),
                "IV": pct(live_best.iv),
                "Volume": int(live_best.volume),
                "Open Interest": int(live_best.open_interest),
            }
            st.table(pd.DataFrame([rec]))

            st.subheader("当前备选合约Top12")
            st.dataframe(
                live_top.style.format(
                    {
                        "strike": "${:.2f}",
                        "premium": "${:.2f}",
                        "annualized_yield": "{:.2%}",
                        "assignment_prob": "{:.2%}",
                        "iv": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )

            st.subheader("最佳策略回测明细")
            st.dataframe(
                bt.style.format(
                    {
                        "spot": "${:.2f}",
                        "trend_shape": "{:.2f}",
                        "delta_target_effective": "{:.2f}",
                        "strike": "${:.2f}",
                        "premium": "${:.2f}",
                        "delta": "{:.2f}",
                        "assignment_prob_model": "{:.2%}",
                        "spot_at_expiry": "${:.2f}",
                        "pnl": "${:.2f}",
                        "return": "{:.2%}",
                        "premium_yield_annualized": "{:.2%}",
                        "equity": "{:.4f}",
                    }
                ),
                use_container_width=True,
            )

            csv = bt.to_csv(index=False).encode("utf-8")
            st.download_button(
                "下载最佳策略回测CSV",
                data=csv,
                file_name=f"{ticker}_8y_sell_call_strategy_backtest.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"执行失败: {e}")


if __name__ == "__main__":
    render_app()

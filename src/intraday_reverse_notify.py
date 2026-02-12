import os
import json
import math
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
from zoneinfo import ZoneInfo  # Python 3.9+

STATE_PATH = "state.json"
JST = timezone(timedelta(hours=9))
NY = ZoneInfo("America/New_York")


# ---------- Slack ----------
def post_slack(webhook_url: str, text: str) -> None:
    r = requests.post(webhook_url, json={"text": text}, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Slack送信失敗: {r.status_code} {r.text}")


# ---------- State ----------
def load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {
            "last_date": "",
            "last_go": False,
            "last_bar_ts": "",
            "pending": False,
            "pending_date": "",
        }
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        st = json.load(f)

    st.setdefault("last_date", "")
    st.setdefault("last_go", False)
    st.setdefault("last_bar_ts", "")
    st.setdefault("pending", False)
    st.setdefault("pending_date", "")
    return st


def save_state(state: dict) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ---------- Helpers ----------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, math.nan)
    return 100 - (100 / (1 + rs))


def now_jst() -> datetime:
    return datetime.now(tz=JST)


def is_run_window_us_market(dt_jst: datetime) -> bool:
    """
    米国市場の通常時間だけ動かす（NY 09:30〜16:10）
    """
    if dt_jst.weekday() >= 5:
        return False

    dt_ny = dt_jst.astimezone(NY)
    h, m = dt_ny.hour, dt_ny.minute
    after_open = (h > 9) or (h == 9 and m >= 30)
    before_close = (h < 16) or (h == 16 and m <= 10)
    return after_open and before_close


def is_morning_digest_time(dt_jst: datetime) -> bool:
    """
    朝まとめ通知の実行時間帯（JST）
    06:05〜06:20 の間だけ true（cronを06:10にしてもOK）
    """
    if dt_jst.weekday() >= 5:
        return False
    return dt_jst.hour == 6 and (5 <= dt_jst.minute <= 20)


def trading_date_ny(ts_utc: pd.Timestamp) -> str:
    dt_ny = ts_utc.to_pydatetime().astimezone(NY)
    return dt_ny.date().isoformat()


# ---------- Data: Finnhub (QQQ 60min) ----------
def fetch_qqq_60m_finnhub(api_key: str) -> pd.DataFrame:
    now = int(datetime.now(timezone.utc).timestamp())
    frm = now - 10 * 24 * 3600

    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        "symbol": "QQQ",
        "resolution": 60,
        "from": frm,
        "to": now,
        "token": api_key,
    }

    try:
        js = requests.get(url, params=params, timeout=20).json()
    except Exception as e:
        print(f"skip: Finnhub request error: {e}")
        return pd.DataFrame()

    if js.get("s") != "ok":
        print(f"skip: Finnhub not ok: s={js.get('s')} msg={js.get('error')}")
        return pd.DataFrame()

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(js["t"], unit="s", utc=True),
        "open": js["o"],
        "high": js["h"],
        "low": js["l"],
        "close": js["c"],
        "volume": js["v"],
    }).sort_values("timestamp").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df["date"] = df["timestamp"].apply(trading_date_ny)
    return df


# ---------- Data: FRED VIX daily ----------
def fetch_fred_vix_daily() -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
    df = pd.read_csv(url)
    df.columns = [str(c).strip() for c in df.columns]

    if "DATE" in df.columns:
        df = df.rename(columns={"DATE": "Date"})
    elif "observation_date" in df.columns:
        df = df.rename(columns={"observation_date": "Date"})
    else:
        raise RuntimeError(f"FRED VIX date列不明 columns={df.columns.tolist()}")

    df = df.rename(columns={"VIXCLS": "Close"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    df["Ret1"] = df["Close"].pct_change() * 100
    return df


# ---------- Signal ----------
def build_intraday_signal(q: pd.DataFrame, vix: pd.DataFrame) -> dict:
    cur_date = q["date"].max()
    prev_dates = sorted(q["date"].unique())
    prev_date = prev_dates[-2] if len(prev_dates) >= 2 else cur_date

    day = q[q["date"] == cur_date].copy().reset_index(drop=True)
    prev_day = q[q["date"] == prev_date].copy().reset_index(drop=True)

    prev_close = float(prev_day.iloc[-1]["close"]) if len(prev_day) else float(day.iloc[0]["open"])

    day["RSI14"] = rsi(day["close"], 14)
    day["MA20"] = day["close"].rolling(20).mean()

    last = day.iloc[-1]
    last_close = float(last["close"])
    last_rsi = float(last["RSI14"]) if not pd.isna(last["RSI14"]) else 50.0
    last_ma20 = float(last["MA20"]) if not pd.isna(last["MA20"]) else float("nan")

    day_low = float(day["low"].min())
    day_open = float(day.iloc[0]["open"])
    ret_from_prev_close = (last_close / prev_close - 1) * 100
    drop_to_low = (day_low / prev_close - 1) * 100
    bounce_from_low = (last_close / day_low - 1) * 100 if day_low > 0 else 0.0

    vix_last = vix.iloc[-1]
    vix_ret1 = float(vix_last["Ret1"]) if not pd.isna(vix_last["Ret1"]) else 0.0
    gate_vix_calm = (vix_ret1 <= 3.0)

    patterns = []

    p1 = (drop_to_low <= -1.0) and (last_rsi <= 35.0) and (bounce_from_low >= 0.3)
    if p1:
        patterns.append("①パニック戻し(60m)")

    gap_down = (day_open <= prev_close * (1 - 0.005))
    reclaim = (last_close >= prev_close * (1 - 0.001))
    if gap_down and reclaim:
        patterns.append("②ギャップ否定(60m)")

    if len(day) >= 4:
        r1 = (day["close"].pct_change() * 100).fillna(0)
        last3_down = (r1.iloc[-4:-1] < 0).all()
        stop_now = (r1.iloc[-1] >= -0.05)
        if bool(last3_down and stop_now):
            patterns.append("③3本目回避(60m)")

    if len(day) >= 6:
        r1 = (day["close"].pct_change() * 100).fillna(0)
        big_drop_seen = (r1.min() <= -0.6)
        flat2 = (abs(r1.iloc[-1]) <= 0.15) and (abs(r1.iloc[-2]) <= 0.15)
        if bool(big_drop_seen and flat2):
            patterns.append("④横ばい圧縮(60m)")

    ma20_ok = (not math.isnan(last_ma20)) and (last_close >= last_ma20)
    go = gate_vix_calm and (len(patterns) > 0)

    return {
        "date": str(cur_date),
        "ts_utc": str(last["timestamp"]),
        "go": go,
        "patterns": patterns,
        "gate_vix_calm": gate_vix_calm,
        "ma20_ok": ma20_ok,
        "prev_close": prev_close,
        "open": day_open,
        "low": day_low,
        "close": last_close,
        "ret_from_prev_close": ret_from_prev_close,
        "drop_to_low": drop_to_low,
        "bounce_from_low": bounce_from_low,
        "rsi14": last_rsi,
        "ma20": last_ma20,
        "vix_close": float(vix_last["Close"]),
        "vix_ret1": vix_ret1,
        "vix_date": vix_last["Date"].strftime("%Y-%m-%d"),
    }


def format_msg(sig: dict) -> str:
    ptxt = " / ".join(sig["patterns"]) if sig["patterns"] else "該当なし"
    strength = "（MA20上=強）" if sig["ma20_ok"] else "（MA20下=弱）"
    msg = (
        f"@here 🟢 逆チャレ GO{strength}\n"
        f"型: {ptxt}\n"
        f"QQQ(60m): date={sig['date']} close={sig['close']:.2f} prevC={sig['prev_close']:.2f}\n"
        f"  drop_to_low={sig['drop_to_low']:.2f}% bounce={sig['bounce_from_low']:.2f}% "
        f"ret={sig['ret_from_prev_close']:.2f}% RSI14={sig['rsi14']:.1f}\n"
        f"VIX(FRED日足): {sig['vix_date']} close={sig['vix_close']:.2f} 1d={sig['vix_ret1']:.2f}% gate={sig['gate_vix_calm']}\n"
        f"運用メモ: 手数料1%前提。+1%即利確 / -0.5〜-1.0%撤退"
    )
    return msg


def main():
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    finnhub_key = os.environ.get("FINNHUB_API_KEY")
    if not webhook:
        raise RuntimeError("SLACK_WEBHOOK_URL が未設定")
    if not finnhub_key:
        raise RuntimeError("FINNHUB_API_KEY が未設定")

    now = now_jst()
    state = load_state()

    # ---------------------------
    # 朝まとめ（06:05〜06:20 JST）
    # ---------------------------
    if is_morning_digest_time(now):
        if not state.get("pending", False):
            print("morning: no pending")
            return

        q = fetch_qqq_60m_finnhub(finnhub_key)
        if q.empty:
            print("morning: no qqq data")
            return

        vix = fetch_fred_vix_daily()
        sig = build_intraday_signal(q, vix)

        # “再判定してまだGOなら送る / 悪化してたら送らない”
        if sig["go"]:
            post_slack(webhook, format_msg(sig))
            print("morning: notified (reconfirmed GO)")
        else:
            print("morning: pending canceled (GO no longer true)")

        # pending を必ず落とす（同じ日に何回も鳴らさない）
        state["pending"] = False
        state["pending_date"] = ""
        save_state(state)
        return

    # ---------------------------
    # 夜間（市場時間中）: 監視して pending を立てるだけ
    # ---------------------------
    if os.environ.get("RUN_ANYTIME", "0") != "1":
        if not is_run_window_us_market(now):
            print("skip: outside US market window")
            return

    q = fetch_qqq_60m_finnhub(finnhub_key)
    if q.empty:
        print("skip: no qqq data (finnhub)")
        return

    # 同じ最新バーならスキップ（無駄打ち防止）
    last_bar_ts = str(q.iloc[-1]["timestamp"])
    if state.get("last_bar_ts", "") == last_bar_ts:
        print(f"skip: same bar {last_bar_ts}")
        return

    vix = fetch_fred_vix_daily()
    sig = build_intraday_signal(q, vix)

    # state更新
    state["last_bar_ts"] = last_bar_ts
    state["last_date"] = sig["date"]
    state["last_go"] = bool(sig["go"])

    # 夜は「候補として pending 立てるだけ」
    # すでにpendingなら維持（何度も上書きしない）
    if sig["go"] and not state.get("pending", False):
        state["pending"] = True
        state["pending_date"] = sig["date"]
        print(f"night: pending set date={sig['date']} ts={sig['ts_utc']} patterns={sig['patterns']}")

    save_state(state)
    print(f"night: done go={sig['go']} pending={state.get('pending')} date={sig['date']}")


if __name__ == "__main__":
    main()

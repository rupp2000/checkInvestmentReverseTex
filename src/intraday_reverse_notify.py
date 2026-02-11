import os
import json
import math
from datetime import datetime, time, timezone, timedelta

import pandas as pd
import requests

STATE_PATH = "state.json"
JST = timezone(timedelta(hours=9))

# ---------- Slack ----------
def post_slack(webhook_url: str, text: str) -> None:
    r = requests.post(webhook_url, json={"text": text}, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Slack送信失敗: {r.status_code} {r.text}")

# ---------- State ----------
def load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {"last_date": "", "last_go": False}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

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

def is_us_session_window_jst(dt: datetime) -> bool:
    """
    米国市場時間“だいたい”に限定してAPIコール数を抑えるガード。
    JSTで:
      夏時間: 22:30-05:00
      冬時間: 23:30-06:00
    をざっくり両対応で広めに取る（22:00-06:30）。
    """
    if dt.weekday() >= 5:  # 土日
        return False
    t = dt.timetz()
    start = time(22, 0, tzinfo=JST)
    end = time(6, 30, tzinfo=JST)
    return (t >= start) or (t <= end)

# ---------- Data: Alpha Vantage (QQQ 15min) ----------
def fetch_qqq_15m_alpha_vantage(api_key: str) -> pd.DataFrame:
    # docs: TIME_SERIES_INTRADAY + interval=15min + outputsize=compact :contentReference[oaicite:3]{index=3}
    url = (
        "https://www.alphavantage.co/query"
        "?function=TIME_SERIES_INTRADAY"
        "&symbol=QQQ"
        "&interval=15min"
        "&outputsize=compact"
        "&datatype=csv"
        f"&apikey={api_key}"
    )
    df = pd.read_csv(url)

    # レート制限時などは列が違う（messageだけ）ことがある
    if "timestamp" not in df.columns:
        raise RuntimeError(f"AlphaVantage応答が想定外 columns={df.columns.tolist()} head={df.head(2).to_dict()}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # 15分足の終値等
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close", "low", "high", "open"])
    # 日付キー（UTC日付でOK。判定の“今日”は最新日付で扱う）
    df["date"] = df["timestamp"].dt.date
    return df

# ---------- Data: FRED VIX daily ----------
def fetch_fred_vix_daily() -> pd.DataFrame:
    # FRED VIXCLS is daily close :contentReference[oaicite:4]{index=4}
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

# ---------- Signal (intraday) ----------
def build_intraday_signal(q: pd.DataFrame, vix: pd.DataFrame) -> dict:
    # 最新日（UTC日付ベース）
    cur_date = q["date"].max()
    prev_dates = sorted(q["date"].unique())
    prev_date = prev_dates[-2] if len(prev_dates) >= 2 else cur_date

    day = q[q["date"] == cur_date].copy().reset_index(drop=True)
    prev_day = q[q["date"] == prev_date].copy().reset_index(drop=True)

    # 前日終値（前日最終バー）
    prev_close = float(prev_day.iloc[-1]["close"]) if len(prev_day) else float(day.iloc[0]["open"])

    # 当日指標
    day["RSI14"] = rsi(day["close"], 14)
    day["MA20"] = day["close"].rolling(20).mean()  # 15分足の20本=約5時間
    last = day.iloc[-1]
    last_close = float(last["close"])
    last_rsi = float(last["RSI14"]) if not pd.isna(last["RSI14"]) else 50.0
    last_ma20 = float(last["MA20"]) if not pd.isna(last["MA20"]) else float("nan")
    day_low = float(day["low"].min())
    day_open = float(day.iloc[0]["open"])
    ret_from_prev_close = (last_close / prev_close - 1) * 100
    drop_to_low = (day_low / prev_close - 1) * 100
    bounce_from_low = (last_close / day_low - 1) * 100 if day_low > 0 else 0.0

    # VIXゲート（日足）：今日のVIXが荒れてる日は抑制
    vix_last = vix.iloc[-1]
    vix_ret1 = float(vix_last["Ret1"]) if not pd.isna(vix_last["Ret1"]) else 0.0
    gate_vix_calm = (vix_ret1 <= 3.0)

    # ---- 逆チャレ（15分足向けに寄せた①〜④）----
    patterns = []

    # ① パニック戻し（当日安値が前日終値比-1.0%以下 ＆ RSI低め ＆ 安値から反発）
    p1 = (drop_to_low <= -1.0) and (last_rsi <= 35.0) and (bounce_from_low >= 0.3)
    if p1:
        patterns.append("①パニック戻し(15m)")

    # ② ギャップ否定（当日寄りが前日終値より下→いま前日終値近くまで回復）
    gap_down = (day_open <= prev_close * (1 - 0.005))      # -0.5%未満で寄り
    reclaim  = (last_close >= prev_close * (1 - 0.001))    # -0.1%以内まで戻す
    p2 = gap_down and reclaim
    if p2:
        patterns.append("②ギャップ否定(15m)")

    # ③ 3本目回避（直近3本が下げ続き→今の足で止まる）
    if len(day) >= 4:
        r1 = (day["close"].pct_change() * 100).fillna(0)
        last3_down = (r1.iloc[-4:-1] < 0).all()
        stop_now = (r1.iloc[-1] >= -0.05)  # 下げ止まり近辺
        p3 = bool(last3_down and stop_now)
        if p3:
            patterns.append("③3本目回避(15m)")

    # ④ 横ばい圧縮（当日大きめ下げが一度出て、その後2本が小動き）
    if len(day) >= 6:
        r1 = (day["close"].pct_change() * 100).fillna(0)
        big_drop_seen = (r1.min() <= -0.6)
        flat2 = (abs(r1.iloc[-1]) <= 0.15) and (abs(r1.iloc[-2]) <= 0.15)
        p4 = bool(big_drop_seen and flat2)
        if p4:
            patterns.append("④横ばい圧縮(15m)")

    # ---- Gate（チャンス増やすため、MA20は“強GO”扱いにして必須化しない）----
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

def main():
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not webhook:
        raise RuntimeError("SLACK_WEBHOOK_URL が未設定")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY が未設定")

    # 無駄打ち防止：米国市場時間っぽい時だけ実行（無料枠対策）
    now = now_jst()
    if os.environ.get("RUN_ANYTIME", "0") != "1":
        if not is_us_session_window_jst(now):
            print("skip: outside US session window (JST)")
            return

    q = fetch_qqq_15m_alpha_vantage(api_key)
    vix = fetch_fred_vix_daily()

    sig = build_intraday_signal(q, vix)

    state = load_state()
    last_date = state.get("last_date", "")
    last_go = bool(state.get("last_go", False))

    # 通知条件：WAIT→GO になった“初回だけ”
    should_notify = (sig["go"] is True) and (last_go is False or last_date != sig["date"])

    # state更新（GO継続でも更新して、同日で2回鳴らない）
    state["last_date"] = sig["date"]
    state["last_go"] = bool(sig["go"])
    save_state(state)

    if not should_notify:
        print(f"no notify: go={sig['go']} last_go={last_go} last_date={last_date} date={sig['date']}")
        return

    ptxt = " / ".join(sig["patterns"]) if sig["patterns"] else "該当なし"
    strength = "（MA20上=強）" if sig["ma20_ok"] else "（MA20下=弱）"

    msg = (
        f"@here 🟢 逆チャレ GO{strength}\n"
        f"型: {ptxt}\n"
        f"QQQ(15m): date={sig['date']} close={sig['close']:.2f} prevC={sig['prev_close']:.2f}\n"
        f"  drop_to_low={sig['drop_to_low']:.2f}% bounce={sig['bounce_from_low']:.2f}% "
        f"ret={sig['ret_from_prev_close']:.2f}% RSI14={sig['rsi14']:.1f}\n"
        f"VIX(FRED日足): {sig['vix_date']} close={sig['vix_close']:.2f} 1d={sig['vix_ret1']:.2f}% gate={sig['gate_vix_calm']}\n"
        f"運用メモ: 手数料1%前提。+1%即利確 / -0.5〜-1.0%撤退"
    )
    post_slack(webhook, msg)

if __name__ == "__main__":
    main()

def check_above_200ma(info, hist):
    """Is price above its 200-day moving average? Long-term bullish."""
    if hist is None or len(hist) < 200:
        return 0, None

    ma_200 = hist["Close"].tail(200).mean()
    current_price = hist["Close"].iloc[-1]

    if current_price > ma_200:
        return 1, f"✅ Price above 200d MA (${ma_200:.2f})"
    return 0, f"❌ Price below 200d MA (${ma_200:.2f})"


def check_golden_cross(info, hist):
    """Has the 50-day MA recently crossed above the 200-day MA? Long-term bullish signal."""
    if hist is None or len(hist) < 200:
        return 0, None

    ma_50_today = hist["Close"].tail(50).mean()
    ma_200_today = hist["Close"].tail(200).mean()

    # check yesterday too to detect a recent cross
    ma_50_yesterday = hist["Close"].tail(51).head(50).mean()
    ma_200_yesterday = hist["Close"].tail(201).head(200).mean()

    # currently in golden cross state (50d above 200d)
    if ma_50_today > ma_200_today:
        # recent cross within last week is extra bullish
        if ma_50_yesterday <= ma_200_yesterday:
            return 1, f"✅ Golden cross just happened (50d: ${ma_50_today:.2f} vs 200d: ${ma_200_today:.2f})"
        return 1, f"✅ 50d MA above 200d MA (bullish trend)"
    return 0, f"❌ 50d MA below 200d MA (bearish trend)"


def check_distance_from_200ma(info, hist):
    """How far below 200d MA? Big drawdowns with strong fundamentals = classic value play."""
    if hist is None or len(hist) < 200:
        return 0, None

    ma_200 = hist["Close"].tail(200).mean()
    current_price = hist["Close"].iloc[-1]
    distance_pct = ((current_price - ma_200) / ma_200) * 100

    # within 5% of 200d MA either direction = healthy
    if -5 <= distance_pct <= 15:
        return 1, f"✅ Near 200d MA ({distance_pct:+.1f}%)"
    # significantly below could be opportunity
    elif distance_pct < -5:
        return 0, f"⚠️ {abs(distance_pct):.1f}% below 200d MA"
    # too far above could be overheated
    else:
        return 0, f"⚠️ {distance_pct:.1f}% above 200d MA (overextended)"
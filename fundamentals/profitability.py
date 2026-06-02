from fundamentals.sector_benchmarks import get_sector_benchmark


def check_operating_margin(info):
    om = info.get("operatingMargins", None)
    sector = info.get("sector", None)
    if om is None:
        return 0, None

    benchmark = get_sector_benchmark(sector, "op_margin_median") if sector else None

    if benchmark is not None:
        # compare to sector median
        if om >= benchmark:
            return 1, f"✅ Operating margin {om * 100:.1f}% beats {sector} median {benchmark * 100:.1f}%"
        return 0, f"❌ Operating margin {om * 100:.1f}% below {sector} median {benchmark * 100:.1f}%"

    # fallback to fixed threshold if no benchmark
    if om > 0.15:
        return 1, f"✅ Strong operating margin {om * 100:.1f}%"
    return 0, f"❌ Weak operating margin {om * 100:.1f}%"


def check_roe(info):
    roe = info.get("returnOnEquity", None)
    sector = info.get("sector", None)
    if roe is None:
        return 0, None

    benchmark = get_sector_benchmark(sector, "roe_median") if sector else None

    if benchmark is not None:
        if roe >= benchmark:
            return 1, f"✅ ROE {roe * 100:.1f}% beats {sector} median {benchmark * 100:.1f}%"
        return 0, f"❌ ROE {roe * 100:.1f}% below {sector} median {benchmark * 100:.1f}%"

    if roe > 0.15:
        return 1, f"✅ Strong ROE {roe * 100:.1f}%"
    return 0, f"❌ Weak ROE {roe * 100:.1f}%"
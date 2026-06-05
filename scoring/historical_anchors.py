"""
Historical Anchors — phase snapshots from real companies for pattern matching.

Each company has 2–3 phase snapshots capturing distinct life stages:
  - survival / crash moments
  - early growth / pivot moments
  - breakout / mania moments
  - peak / decline moments (for losers)

The bot matches current candidates against these patterns using cosine
similarity on fundamental fingerprints — bucket-aware by default.

Fields per phase:
    year:           int    — fiscal year of snapshot
    bucket:         str    — which scoring bucket this anchor belongs to
    revenue:        float  — total revenue (USD)
    revenue_growth: float  — YoY revenue growth (decimal, e.g. 0.20 = 20%)
    gross_margin:   float  — gross profit / revenue (decimal)
    op_margin:      float  — operating income / revenue (decimal)
    fcf_margin:     float  — free cash flow / revenue (decimal, can be negative)
    net_margin:     float  — net income / revenue (decimal, can be negative)
    macro_context:  str    — short tag for macro regime
    outcome:        str    — "winner" | "loser" | "mixed"
    notes:          str    — qualitative context, what was happening
"""

ANCHORS = {

    # =========================================================
    # AAPL — Apple Inc.
    # Arc: near-bankruptcy → iPod recovery → iPhone mania
    # =========================================================
    "AAPL": {
        "1997-survival-near-bankruptcy": {
            "year": 1997,
            "bucket": "mature_tech",
            "revenue": 7.1e9,
            "revenue_growth": -0.28,
            "gross_margin": 0.19,
            "op_margin": -0.14,
            "fcf_margin": -0.05,
            "net_margin": -0.15,
            "macro_context": "post-tech-recession",
            "outcome": "winner",
            "notes": "90 days from bankruptcy per Steve Jobs. Jobs returns. Microsoft $150M lifeline.",
        },
        "2003-ipod-era-recovery": {
            "year": 2003,
            "bucket": "mature_tech",
            "revenue": 6.21e9,
            "revenue_growth": 0.08,
            "gross_margin": 0.275,
            "op_margin": 0.005,
            "fcf_margin": 0.05,
            "net_margin": 0.011,
            "macro_context": "low-rates-post-bubble",
            "outcome": "winner",
            "notes": "Barely profitable. iPod just hitting. Pre-iPhone. Revenue still below 2002.",
        },
        "2010-iphone-mania-breakout": {
            "year": 2010,
            "bucket": "mature_tech",
            "revenue": 65.2e9,
            "revenue_growth": 0.52,
            "gross_margin": 0.40,
            "op_margin": 0.28,
            "fcf_margin": 0.27,
            "net_margin": 0.22,
            "macro_context": "post-gfc-zirp",
            "outcome": "winner",
            "notes": "iPhone hits critical mass + iPad launched. Revenue exploded. Cash machine emerging.",
        },
    },

    # =========================================================
    # MSFT — Microsoft
    # Arc: stuck post-bubble → Satya pivot → cloud machine
    # =========================================================
    "MSFT": {
        "2003-stuck-post-bubble": {
            "year": 2003,
            "bucket": "mature_tech",
            "revenue": 32.19e9,
            "revenue_growth": 0.13,
            "gross_margin": 0.85,
            "op_margin": 0.41,
            "fcf_margin": 0.42,
            "net_margin": 0.31,
            "macro_context": "post-dot-com",
            "outcome": "winner",
            "notes": "Peak software era but worried about Linux/open source. Stock went sideways for a decade.",
        },
        "2014-satya-cloud-pivot": {
            "year": 2014,
            "bucket": "mature_tech",
            "revenue": 86.83e9,
            "revenue_growth": 0.12,
            "gross_margin": 0.69,
            "op_margin": 0.32,
            "fcf_margin": 0.30,
            "net_margin": 0.25,
            "macro_context": "early-zirp",
            "outcome": "winner",
            "notes": "Nadella takes over Feb 2014. Windows mobile failing. 'Mobile-first cloud-first' pivot announced.",
        },
        "2018-cloud-machine-breakout": {
            "year": 2018,
            "bucket": "mature_tech",
            "revenue": 110.36e9,
            "revenue_growth": 0.14,
            "gross_margin": 0.65,
            "op_margin": 0.32,
            "fcf_margin": 0.29,
            "net_margin": 0.15,  # GAAP — non-GAAP was ~27%; TCJA one-time charge
            "macro_context": "peak-zirp-post-tax-reform",
            "outcome": "winner",
            "notes": "First time crossing $100B revenue. Azure +98% YoY. Market cap doubled. Transformation complete.",
        },
    },

    # =========================================================
    # GOOGL — Alphabet
    # Arc: mobile doubt → mobile mature → AI existential fear
    # =========================================================
    "GOOGL": {
        "2008-mobile-transition-doubt": {
            "year": 2008,
            "bucket": "mature_tech",
            "revenue": 21.8e9,
            "revenue_growth": 0.31,
            "gross_margin": 0.60,
            "op_margin": 0.30,
            "fcf_margin": 0.27,
            "net_margin": 0.19,
            "macro_context": "gfc",
            "outcome": "winner",
            "notes": "iPhone launched 2007. Investors feared mobile would kill search profitability. Survived and dominated.",
        },
        "2015-mobile-search-maturation": {
            "year": 2015,
            "bucket": "mature_tech",
            "revenue": 74.99e9,
            "revenue_growth": 0.14,
            "gross_margin": 0.62,
            "op_margin": 0.26,
            "fcf_margin": 0.21,
            "net_margin": 0.22,
            "macro_context": "post-zirp",
            "outcome": "winner",
            "notes": "Sundar Pichai takes over Aug 2015. Mobile search proven. Ad model adapted brilliantly.",
        },
        "2022-chatgpt-existential-fear": {
            "year": 2022,
            "bucket": "mature_tech",
            "revenue": 282.84e9,
            "revenue_growth": 0.10,
            "gross_margin": 0.55,
            "op_margin": 0.26,
            "fcf_margin": 0.21,
            "net_margin": 0.21,
            "macro_context": "rate-hikes",
            "outcome": "winner",
            "notes": "ChatGPT launched Nov 2022 → GOOGL -45% from peak on AI disruption fears. Rallied back proving panic overdone.",
        },
    },

    # =========================================================
    # AMZN — Amazon
    # Arc: dot-com crash → post-recession scale → AWS reveal
    # =========================================================
    "AMZN": {
        "2001-dotcom-crash-survival": {
            "year": 2001,
            "bucket": "consumer_cyclical",
            "revenue": 3.12e9,
            "revenue_growth": 0.13,
            "gross_margin": 0.26,
            "op_margin": -0.13,
            "fcf_margin": -0.06,
            "net_margin": -0.18,
            "macro_context": "dot-com-crash",
            "outcome": "winner",
            "notes": "Stock fell from $113 to $5.97 (-94%). Many called it dead. Bezos focused on cash flow.",
        },
        "2008-post-recession-scale": {
            "year": 2008,
            "bucket": "consumer_cyclical",
            "revenue": 19.17e9,
            "revenue_growth": 0.29,
            "gross_margin": 0.22,
            "op_margin": 0.044,
            "fcf_margin": 0.06,
            "net_margin": 0.034,
            "macro_context": "gfc",
            "outcome": "winner",
            "notes": "Cash machine emerging. Prime launched 2005, Kindle 2007. Profitable but thin margins.",
        },
        "2015-aws-profit-reveal": {
            "year": 2015,
            "bucket": "consumer_cyclical",
            "revenue": 107.01e9,
            "revenue_growth": 0.20,
            "gross_margin": 0.33,
            "op_margin": 0.021,
            "fcf_margin": 0.08,
            "net_margin": 0.006,
            "macro_context": "post-zirp",
            "outcome": "winner",
            "notes": "First AWS segment disclosure: $7.9B at ~17% op margin. Stock more than doubled. Hidden gem revealed.",
        },
    },

    # =========================================================
    # META — Meta Platforms (Facebook)
    # Arc: Cambridge Analytica crisis → metaverse crash → AI pivot
    # =========================================================
    "META": {
        "2018-cambridge-analytica-crisis": {
            "year": 2018,
            "bucket": "mature_tech",
            "revenue": 55.84e9,
            "revenue_growth": 0.37,
            "gross_margin": 0.83,
            "op_margin": 0.45,
            "fcf_margin": 0.27,
            "net_margin": 0.40,
            "macro_context": "peak-zirp",
            "outcome": "winner",
            "notes": "Cambridge Analytica scandal mid-2018. Regulatory threat. Stock -40% from peak but fundamentals stayed strong.",
        },
        "2022-metaverse-crash": {
            "year": 2022,
            "bucket": "mature_tech",
            "revenue": 116.61e9,
            "revenue_growth": -0.01,
            "gross_margin": 0.78,
            "op_margin": 0.25,
            "fcf_margin": 0.16,
            "net_margin": 0.20,
            "macro_context": "rate-hikes",
            "outcome": "winner",
            "notes": "Reality Labs lost $13.7B. Ad business hit by Apple ATT + TikTok. Stock -76% peak-to-trough.",
        },
        "2024-ai-pivot-recovery": {
            "year": 2024,
            "bucket": "mature_tech",
            "revenue": 164.50e9,
            "revenue_growth": 0.22,
            "gross_margin": 0.82,
            "op_margin": 0.42,
            "fcf_margin": 0.32,
            "net_margin": 0.38,
            "macro_context": "ai-mania",
            "outcome": "winner",
            "notes": "Year of Efficiency (2023) → Llama AI integration (2024) → ad targeting revival. Stock fully recovered.",
        },
    },

    # =========================================================
    # CRM — Salesforce
    # Arc: early profitability → financial crisis test → pandemic acceleration
    # =========================================================
    "CRM": {
        "2008-early-profitability": {
            "year": 2008,
            "bucket": "saas_growth",
            "revenue": 0.749e9,
            "revenue_growth": 0.51,
            "gross_margin": 0.78,
            "op_margin": 0.03,
            "fcf_margin": 0.30,
            "net_margin": 0.025,
            "macro_context": "pre-gfc",
            "outcome": "winner",
            "notes": "Fiscal year ended Jan 2008. First SaaS company at $850M run rate. Just turned profitable.",
        },
        "2009-recession-resilience": {
            "year": 2009,
            "bucket": "saas_growth",
            "revenue": 1.077e9,
            "revenue_growth": 0.44,
            "gross_margin": 0.78,
            "op_margin": 0.06,
            "fcf_margin": 0.28,
            "net_margin": 0.04,
            "macro_context": "gfc",
            "outcome": "winner",
            "notes": "Through GFC, growth slowed but stayed strong. Proved SaaS subscriptions defensive vs enterprise licenses.",
        },
        "2020-pandemic-acceleration": {
            "year": 2020,
            "bucket": "saas_growth",
            "revenue": 17.10e9,
            "revenue_growth": 0.29,
            "gross_margin": 0.75,
            "op_margin": 0.02,
            "fcf_margin": 0.25,
            "net_margin": 0.07,
            "macro_context": "pandemic-zirp",
            "outcome": "winner",
            "notes": "Fiscal year ended Jan 2020. Digital transformation surge. Stock hit ATH in 2021 at ~$310 then fell -60%.",
        },
    },

    # =========================================================
    # ADBE — Adobe
    # Arc: subscription pivot start → mature SaaS → ChatGPT/Figma fear
    # =========================================================
    "ADBE": {
        "2012-creative-cloud-pivot": {
            "year": 2012,
            "bucket": "saas_growth",
            "revenue": 4.40e9,
            "revenue_growth": 0.04,
            "gross_margin": 0.88,
            "op_margin": 0.26,
            "fcf_margin": 0.27,
            "net_margin": 0.19,
            "macro_context": "post-gfc-zirp",
            "outcome": "winner",
            "notes": "Creative Cloud launched. Painful transition: perpetual revenue cut to recognize subscription over time. Margin compression scared investors.",
        },
        "2018-subscription-machine": {
            "year": 2018,
            "bucket": "saas_growth",
            "revenue": 9.03e9,
            "revenue_growth": 0.24,
            "gross_margin": 0.86,
            "op_margin": 0.31,
            "fcf_margin": 0.39,
            "net_margin": 0.29,
            "macro_context": "peak-zirp",
            "outcome": "winner",
            "notes": "Subscription transition complete. Revenue more than doubled from 2012. Margins recovered. Multiple expansion.",
        },
        "2021-creative-cloud-peak": {
            "year": 2021,
            "bucket": "saas_growth",
            "revenue": 15.79e9,
            "revenue_growth": 0.23,
            "gross_margin": 0.88,
            "op_margin": 0.37,
            "fcf_margin": 0.45,
            "net_margin": 0.31,
            "macro_context": "pandemic-zirp",
            "outcome": "winner",
            "notes": "Peak ZIRP valuation. Stock $700. Then 2022-23 Figma deal blocked + AI fears (Midjourney) → -50% drawdown.",
        },
    },

    # =========================================================
    # NOW — ServiceNow
    # Arc: hyper-growth unprofitable → profitability inflection → mature compounder
    # =========================================================
    "NOW": {
        "2016-hyper-growth-unprofitable": {
            "year": 2016,
            "bucket": "saas_growth",
            "revenue": 1.39e9,
            "revenue_growth": 0.38,
            "gross_margin": 0.71,
            "op_margin": -0.08,
            "fcf_margin": 0.21,
            "net_margin": -0.16,
            "macro_context": "low-rates",
            "outcome": "winner",
            "notes": "Classic high-growth SaaS shape. GAAP losses but strong FCF. Rule of 40 = 38 + 21 = 59. Investors trusted the model.",
        },
        "2018-rule-of-40-king": {
            "year": 2018,
            "bucket": "saas_growth",
            "revenue": 2.61e9,
            "revenue_growth": 0.36,
            "gross_margin": 0.76,
            "op_margin": -0.03,
            "fcf_margin": 0.27,
            "net_margin": -0.02,
            "macro_context": "peak-zirp",
            "outcome": "winner",
            "notes": "Approaching breakeven on GAAP. Rule of 40 = 63. Workflow expansion beyond IT into HR, finance.",
        },
        "2021-breakout-profitability": {
            "year": 2021,
            "bucket": "saas_growth",
            "revenue": 5.90e9,
            "revenue_growth": 0.30,
            "gross_margin": 0.78,
            "op_margin": 0.04,
            "fcf_margin": 0.31,
            "net_margin": 0.04,
            "macro_context": "pandemic-zirp",
            "outcome": "winner",
            "notes": "First fully profitable year on GAAP. Stock hit ATH 2021 ~$700, then -45% in 2022 rate hikes, then full recovery.",
        },
    },

    # =========================================================
    # NVDA — Nvidia
    # Arc: GFC near-miss → AI seeding → AI mania
    # =========================================================
    "NVDA": {
        "2009-gfc-near-loss": {
            "year": 2009,
            "bucket": "semiconductors",
            "revenue": 3.43e9,
            "revenue_growth": -0.16,
            "gross_margin": 0.35,
            "op_margin": -0.04,
            "fcf_margin": 0.02,
            "net_margin": -0.01,
            "macro_context": "gfc",
            "outcome": "winner",
            "notes": "Fiscal year ended Jan 2009. Defective chip charges + recession. Stock under $10. Still investing in CUDA — paid off later.",
        },
        "2017-ai-seeds-planted": {
            "year": 2017,
            "bucket": "semiconductors",
            "revenue": 6.91e9,
            "revenue_growth": 0.38,
            "gross_margin": 0.59,
            "op_margin": 0.27,
            "fcf_margin": 0.23,
            "net_margin": 0.24,
            "macro_context": "low-rates",
            "outcome": "winner",
            "notes": "Fiscal year ended Jan 2017. Deep learning + crypto demand. Data center +145%. Stock 10x'd 2016-2018. Pre-AI-mania.",
        },
        "2024-ai-mania-breakout": {
            "year": 2024,
            "bucket": "semiconductors",
            "revenue": 60.92e9,
            "revenue_growth": 1.26,
            "gross_margin": 0.74,
            "op_margin": 0.61,
            "fcf_margin": 0.46,
            "net_margin": 0.49,
            "macro_context": "ai-mania",
            "outcome": "winner",
            "notes": "Fiscal year ended Jan 2024. Data center $47.5B (+217%). H100 monopoly. Most extraordinary fundamental shape in semi history.",
        },
    },

    # =========================================================
    # AMD — Advanced Micro Devices
    # Arc: near bankruptcy → Ryzen turnaround → AI competitor
    # =========================================================
    "AMD": {
        "2015-near-bankruptcy": {
            "year": 2015,
            "bucket": "semiconductors",
            "revenue": 3.99e9,
            "revenue_growth": -0.28,
            "gross_margin": 0.27,
            "op_margin": -0.12,
            "fcf_margin": -0.05,
            "net_margin": -0.17,
            "macro_context": "low-rates",
            "outcome": "winner",
            "notes": "Stock under $3. Bankruptcy fears. Lisa Su CEO (Oct 2014). 4th straight loss year. $2B debt. Refocused on high-perf computing.",
        },
        "2018-ryzen-turnaround": {
            "year": 2018,
            "bucket": "semiconductors",
            "revenue": 6.48e9,
            "revenue_growth": 0.23,
            "gross_margin": 0.38,
            "op_margin": 0.07,
            "fcf_margin": 0.05,
            "net_margin": 0.05,
            "macro_context": "peak-zirp",
            "outcome": "winner",
            "notes": "Ryzen + EPYC ramping. First full profitable year since 2011. Stock 10x'd 2016-2018. Intel stumbled on 10nm.",
        },
        "2024-ai-challenger": {
            "year": 2024,
            "bucket": "semiconductors",
            "revenue": 25.79e9,
            "revenue_growth": 0.14,
            "gross_margin": 0.51,
            "op_margin": 0.08,
            "fcf_margin": 0.10,
            "net_margin": 0.06,
            "macro_context": "ai-mania",
            "outcome": "winner",
            "notes": "MI300 launched late 2023. Data center revenue >50% of total. Distant #2 to NVDA but real AI traction. Stock volatile +/- 40%.",
        },
    },

    # =========================================================
    # MU — Micron Technology
    # Arc: classic memory cycle: trough → peak → AI-era trough
    # Note: cycle stock — these are 'phases' of a cycle, NOT a linear winner story
    # =========================================================
    "MU": {
        "2016-dram-cycle-trough": {
            "year": 2016,
            "bucket": "semiconductors",
            "revenue": 12.40e9,
            "revenue_growth": -0.23,
            "gross_margin": 0.18,
            "op_margin": -0.04,
            "fcf_margin": 0.04,
            "net_margin": -0.02,
            "macro_context": "low-rates-memory-glut",
            "outcome": "mixed",
            "notes": "Fiscal year ended Sep 2016. DRAM price crash, NAND oversupply. Workforce cut. Stock $10. Classic cycle bottom.",
        },
        "2018-dram-cycle-peak": {
            "year": 2018,
            "bucket": "semiconductors",
            "revenue": 30.39e9,
            "revenue_growth": 0.50,
            "gross_margin": 0.59,
            "op_margin": 0.50,
            "fcf_margin": 0.45,
            "net_margin": 0.47,
            "macro_context": "peak-zirp-data-center-boom",
            "outcome": "mixed",
            "notes": "Fiscal year ended Aug 2018. Record memory cycle peak. Stock $60+. CAREFUL: every cycle peak looks unstoppable until it isn't.",
        },
        "2023-ai-era-trough": {
            "year": 2023,
            "bucket": "semiconductors",
            "revenue": 15.54e9,
            "revenue_growth": -0.49,
            "gross_margin": -0.09,
            "op_margin": -0.37,
            "fcf_margin": -0.50,
            "net_margin": -0.38,
            "macro_context": "rate-hikes",
            "outcome": "mixed",
            "notes": "Fiscal year ended Aug 2023. Massive losses. But HBM/AI demand inflection coming. Stock bottomed ~$50, doubled by 2024.",
        },
    },

    # =========================================================
    # LLY — Eli Lilly
    # Arc: patent cliff stagnation → recovery launches → GLP-1 explosion
    # =========================================================
    "LLY": {
        "2017-patent-cliff-stagnation": {
            "year": 2017,
            "bucket": "pharma",
            "revenue": 22.87e9,
            "revenue_growth": 0.08,
            "gross_margin": 0.74,
            "op_margin": 0.14,
            "fcf_margin": 0.16,
            "net_margin": -0.09,  # TCJA charge — non-GAAP positive
            "macro_context": "peak-zirp",
            "outcome": "winner",
            "notes": "Stagnation period before launches. Loss of Cymbalta exclusivity. Stock $75-85, sideways for years.",
        },
        "2020-launch-portfolio-recovery": {
            "year": 2020,
            "bucket": "pharma",
            "revenue": 24.54e9,
            "revenue_growth": 0.10,
            "gross_margin": 0.77,
            "op_margin": 0.27,
            "fcf_margin": 0.24,
            "net_margin": 0.25,
            "macro_context": "pandemic-zirp",
            "outcome": "winner",
            "notes": "Trulicity, Taltz, Verzenio ramping. Pipeline value emerging. Tirzepatide (Mounjaro) in late trials.",
        },
        "2024-glp1-explosion": {
            "year": 2024,
            "bucket": "pharma",
            "revenue": 45.04e9,
            "revenue_growth": 0.32,
            "gross_margin": 0.81,
            "op_margin": 0.32,
            "fcf_margin": 0.20,
            "net_margin": 0.23,
            "macro_context": "rate-hikes-glp1-mania",
            "outcome": "winner",
            "notes": "Zepbound + Mounjaro combined >50% of revenue. Stock 6x'd 2021-2024. Trillion-dollar market cap.",
        },
    },

    # =========================================================
    # REGN — Regeneron
    # Arc: pre-Eylea unprofitable → Eylea launch → Dupixent/Eylea machine
    # =========================================================
    "REGN": {
        "2013-eylea-launch-breakout": {
            "year": 2013,
            "bucket": "biotech",
            "revenue": 2.10e9,
            "revenue_growth": 0.41,
            "gross_margin": 0.84,
            "op_margin": 0.34,
            "fcf_margin": 0.30,
            "net_margin": 0.27,
            "macro_context": "low-rates",
            "outcome": "winner",
            "notes": "Eylea launch (2011) drove first profitable years. Stock 5x'd 2011-2013. Classic biotech turning the corner.",
        },
        "2020-dupixent-eylea-machine": {
            "year": 2020,
            "bucket": "biotech",
            "revenue": 8.50e9,
            "revenue_growth": 0.30,
            "gross_margin": 0.89,
            "op_margin": 0.41,
            "fcf_margin": 0.42,
            "net_margin": 0.42,
            "macro_context": "pandemic-zirp",
            "outcome": "winner",
            "notes": "Dupixent global $3.5B (+70%). Plus REGEN-COV COVID antibody. Operating cash machine in biotech.",
        },
        "2024-eylea-cliff-fears": {
            "year": 2024,
            "bucket": "biotech",
            "revenue": 14.20e9,
            "revenue_growth": 0.08,
            "gross_margin": 0.88,
            "op_margin": 0.28,
            "fcf_margin": 0.32,
            "net_margin": 0.30,
            "macro_context": "rate-hikes",
            "outcome": "winner",
            "notes": "Eylea biosimilars entering. EYLEA HD transition. Stock -40%. Pipeline questions emerging.",
        },
    },

    # =========================================================
    # JPM — JPMorgan Chase
    # Arc: GFC survival → post-DFA mature → ZIRP-era peak
    # =========================================================
    "JPM": {
        "2009-gfc-survival": {
            "year": 2009,
            "bucket": "bank",
            "revenue": 108.6e9,
            "revenue_growth": 0.49,  # versus 2008 base
            "gross_margin": None,  # banks don't have meaningful gross margin
            "op_margin": 0.20,
            "fcf_margin": None,
            "net_margin": 0.11,
            "macro_context": "gfc",
            "outcome": "winner",
            "notes": "Stayed profitable through GFC. One of two big US banks to do so (with WFC). $11.7B net income, ROE 6%. Dimon proved best-of-breed.",
        },
        "2018-mature-bank-restored": {
            "year": 2018,
            "bucket": "bank",
            "revenue": 109.0e9,
            "revenue_growth": 0.09,
            "gross_margin": None,
            "op_margin": 0.40,
            "fcf_margin": None,
            "net_margin": 0.30,
            "macro_context": "rate-hike-cycle-tcja",
            "outcome": "winner",
            "notes": "Net income $32.5B, ROE 13%. Post-DFA regulatory burden integrated. Steady compounder. Stock $90-110.",
        },
        "2021-zirp-bank-peak": {
            "year": 2021,
            "bucket": "bank",
            "revenue": 121.6e9,
            "revenue_growth": 0.01,
            "gross_margin": None,
            "op_margin": 0.49,
            "fcf_margin": None,
            "net_margin": 0.40,
            "macro_context": "pandemic-zirp",
            "outcome": "winner",
            "notes": "Record net income $48.3B. ROE 19%. Stock $170. Reserve releases boosted. Dimon: 'not core or recurring profits'.",
        },
    },

    # =========================================================
    # BAC — Bank of America
    # Arc: post-Countrywide disaster → Moynihan repair → restored compounder
    # =========================================================
    "BAC": {
        "2010-countrywide-disaster": {
            "year": 2010,
            "bucket": "bank",
            "revenue": 110.2e9,
            "revenue_growth": -0.07,
            "gross_margin": None,
            "op_margin": -0.02,
            "fcf_margin": None,
            "net_margin": -0.02,
            "macro_context": "post-gfc",
            "outcome": "winner",
            "notes": "$2.2B net LOSS. $12.4B goodwill writedown. Countrywide litigation. Stock $5-12. 'Necessary repair year' per Moynihan.",
        },
        "2015-turnaround-confirmed": {
            "year": 2015,
            "bucket": "bank",
            "revenue": 82.9e9,
            "revenue_growth": -0.04,
            "gross_margin": None,
            "op_margin": 0.22,
            "fcf_margin": None,
            "net_margin": 0.18,
            "macro_context": "low-rates-zirp-end",
            "outcome": "winner",
            "notes": "First post-crisis 'normal' year. Stock had doubled from 2011 lows. Legacy assets mostly resolved.",
        },
        "2021-zirp-recovery-peak": {
            "year": 2021,
            "bucket": "bank",
            "revenue": 89.1e9,
            "revenue_growth": 0.04,
            "gross_margin": None,
            "op_margin": 0.42,
            "fcf_margin": None,
            "net_margin": 0.36,
            "macro_context": "pandemic-zirp",
            "outcome": "winner",
            "notes": "Net income $32B. ROE 12%. Stock $48 ATH. Largest beneficiary of rate hike cycle expectations.",
        },
    },

    # =========================================================
    # CMG — Chipotle
    # Arc: E.coli crisis → margin recovery struggle → Niccol breakout
    # =========================================================
    "CMG": {
        "2015-ecoli-crisis": {
            "year": 2015,
            "bucket": "consumer_cyclical",
            "revenue": 4.50e9,
            "revenue_growth": 0.10,
            "gross_margin": 0.27,
            "op_margin": 0.18,
            "fcf_margin": 0.16,
            "net_margin": 0.10,
            "macro_context": "low-rates",
            "outcome": "winner",
            "notes": "Late 2015 E. coli outbreak. Stock $750 → $250 by 2018. Margins crashed but core franchise survived.",
        },
        "2018-recovery-attempt": {
            "year": 2018,
            "bucket": "consumer_cyclical",
            "revenue": 4.86e9,
            "revenue_growth": 0.085,
            "gross_margin": 0.27,
            "op_margin": 0.06,
            "fcf_margin": 0.07,
            "net_margin": 0.04,
            "macro_context": "peak-zirp",
            "outcome": "winner",
            "notes": "Brian Niccol (ex-Taco Bell) named CEO March 2018. Stock turning. Digital strategy launching.",
        },
        "2022-niccol-breakout": {
            "year": 2022,
            "bucket": "consumer_cyclical",
            "revenue": 8.63e9,
            "revenue_growth": 0.144,
            "gross_margin": 0.36,
            "op_margin": 0.14,
            "fcf_margin": 0.13,
            "net_margin": 0.10,
            "macro_context": "rate-hikes",
            "outcome": "winner",
            "notes": "Full margin recovery + record store openings + digital. Stock $1700+ ATH. Crisis-to-comeback complete.",
        },
    },

    # =========================================================
    # DPZ — Domino's Pizza
    # Arc: turnaround relaunch → digital domination → mature compounder
    # =========================================================
    "DPZ": {
        "2010-recipe-relaunch": {
            "year": 2010,
            "bucket": "consumer_cyclical",
            "revenue": 1.57e9,
            "revenue_growth": 0.115,
            "gross_margin": 0.27,
            "op_margin": 0.16,
            "fcf_margin": 0.11,
            "net_margin": 0.05,
            "macro_context": "post-gfc",
            "outcome": "winner",
            "notes": "Honest 'our pizza tasted bad' admission + recipe relaunch. Stock $12. The decade's best-performing stock began here.",
        },
        "2015-digital-domination": {
            "year": 2015,
            "bucket": "consumer_cyclical",
            "revenue": 2.22e9,
            "revenue_growth": 0.10,
            "gross_margin": 0.30,
            "op_margin": 0.18,
            "fcf_margin": 0.13,
            "net_margin": 0.08,
            "macro_context": "low-rates",
            "outcome": "winner",
            "notes": "AnyWare ordering platform. Stock $120 (+1000% from 2010). Tech-enabled QSR thesis playing out.",
        },
        "2021-pandemic-peak": {
            "year": 2021,
            "bucket": "consumer_cyclical",
            "revenue": 4.36e9,
            "revenue_growth": 0.10,
            "gross_margin": 0.39,
            "op_margin": 0.18,
            "fcf_margin": 0.10,
            "net_margin": 0.10,
            "macro_context": "pandemic-zirp",
            "outcome": "winner",
            "notes": "Pandemic delivery boom. Stock $560 ATH. Then -45% as comps lapped and delivery aggregators competed.",
        },
    },

    # =========================================================
    # CSCO — Cisco Systems
    # Arc: dot-com bubble peak → permanent re-rating → never came back
    # CAUTION: this is a LOSER pattern — financials looked perfect at the peak
    # =========================================================
    "CSCO": {
        "2000-dotcom-bubble-peak": {
            "year": 2000,
            "bucket": "mature_tech",
            "revenue": 18.93e9,
            "revenue_growth": 0.56,
            "gross_margin": 0.65,
            "op_margin": 0.265,
            "fcf_margin": 0.27,
            "net_margin": 0.14,
            "macro_context": "dot-com-bubble-peak",
            "outcome": "loser",
            "notes": "Briefly most valuable company in world ($580B). P/E ~200. Stock $80. Then -86% over 18 months. WARNING PATTERN.",
        },
        "2002-post-crash-bottom": {
            "year": 2002,
            "bucket": "mature_tech",
            "revenue": 18.92e9,
            "revenue_growth": -0.15,
            "gross_margin": 0.61,
            "op_margin": 0.20,
            "fcf_margin": 0.30,
            "net_margin": 0.09,
            "macro_context": "post-dot-com",
            "outcome": "loser",
            "notes": "Revenue flat-ish, fundamentals OK — but valuation crushed. Stock $8. NEVER reclaimed 2000 high (still hasn't 25 years later).",
        },
        "2010-mature-compounder": {
            "year": 2010,
            "bucket": "mature_tech",
            "revenue": 40.0e9,
            "revenue_growth": 0.11,
            "gross_margin": 0.63,
            "op_margin": 0.25,
            "fcf_margin": 0.25,
            "net_margin": 0.20,
            "macro_context": "low-rates",
            "outcome": "loser",
            "notes": "Doubled revenue from 2000 — but stock $24. Permanently re-rated low. The 'great company, terrible investment' pattern.",
        },
    },

    # =========================================================
    # INTC — Intel
    # Arc: peak dominance → technology lag → market share loss
    # CAUTION: classic incumbent disruption — slow-motion decline
    # =========================================================
    "INTC": {
        "2018-peak-dominance": {
            "year": 2018,
            "bucket": "semiconductors",
            "revenue": 70.85e9,
            "revenue_growth": 0.13,
            "gross_margin": 0.62,
            "op_margin": 0.32,
            "fcf_margin": 0.20,
            "net_margin": 0.29,
            "macro_context": "peak-zirp-tax-reform",
            "outcome": "loser",
            "notes": "Looked invincible — record revenue, 10nm just had to ship. 7nm was the real issue. Stock $50 — never sustained back there.",
        },
        "2021-fab-trouble-emerges": {
            "year": 2021,
            "bucket": "semiconductors",
            "revenue": 79.02e9,
            "revenue_growth": 0.018,
            "gross_margin": 0.55,
            "op_margin": 0.246,
            "fcf_margin": 0.12,
            "net_margin": 0.25,
            "macro_context": "pandemic-zirp",
            "outcome": "loser",
            "notes": "Revenue peak. Gelsinger took over (Feb 2021). Foundry strategy announced. Margins already eroding. AMD share gains accelerating.",
        },
        "2022-collapse": {
            "year": 2022,
            "bucket": "semiconductors",
            "revenue": 63.05e9,
            "revenue_growth": -0.20,
            "gross_margin": 0.43,
            "op_margin": 0.037,
            "fcf_margin": -0.20,
            "net_margin": 0.13,
            "macro_context": "rate-hikes",
            "outcome": "loser",
            "notes": "Revenue -20%. Op margin from 25% → 4%. Negative FCF. Stock -50%. Dividend cut 2023. Total disruption confirmed.",
        },
        "2025-foundry-recovery": {
            "year": 2025,
            "bucket": "semiconductors",
            "revenue": 52.90e9,
            "revenue_growth": 0.0,
            "gross_margin": 0.38,
            "op_margin": 0.05,
            "fcf_margin": -0.10,
            "net_margin": 0.02,
            "macro_context": "ai-mania",
            "outcome": "winner",
            "notes": "18A node shipping (Panther Lake). NVIDIA invested $5B. Lip-Bu Tan CEO. Stock $17 low → $48. Five quarters beating guidance. Real turnaround begun.",
        },
    },

    # =========================================================
    # GE — General Electric
    # Arc: complacent conglomerate → leadership crisis → multi-year collapse
    # CAUTION: classic overlevered industrial — looked like a blue chip
    # =========================================================
    "GE": {
        "2015-pre-crisis-conglomerate": {
            "year": 2015,
            "bucket": "industrial",
            "revenue": 117.0e9,
            "revenue_growth": -0.02,
            "gross_margin": 0.26,
            "op_margin": 0.10,
            "fcf_margin": 0.16,
            "net_margin": -0.05,  # had losses related to GE Capital exit
            "macro_context": "low-rates",
            "outcome": "loser",
            "notes": "Pre-implosion. Stock $30. Power division Alstom acquisition just completed. Looked like turnaround story but rot underneath.",
        },
        "2017-immelt-out-flannery-in": {
            "year": 2017,
            "bucket": "industrial",
            "revenue": 122.1e9,
            "revenue_growth": 0.043,
            "gross_margin": 0.22,
            "op_margin": 0.05,
            "fcf_margin": 0.08,
            "net_margin": -0.05,
            "macro_context": "peak-zirp",
            "outcome": "loser",
            "notes": "Immelt out August 2017. Flannery slashed dividend 50%. Stock -45%. Power division cash flow collapse revealed.",
        },
        "2018-collapse-confirmed": {
            "year": 2018,
            "bucket": "industrial",
            "revenue": 121.6e9,
            "revenue_growth": -0.004,
            "gross_margin": 0.20,
            "op_margin": -0.18,  # $22B goodwill writedown
            "fcf_margin": 0.04,
            "net_margin": -0.18,
            "macro_context": "rate-hikes",
            "outcome": "loser",
            "notes": "$22B Power goodwill writedown. Stock -56% (after -45% in 2017). Total -76% over 2 years. Culp brought in from outside.",
        },
    },

    # =========================================================
    # IBM — International Business Machines
    # Arc: peak revenue → 22-quarter decline → cloud transition (still struggling)
    # CAUTION: classic slow-motion incumbent decline. Stock looked like value forever.
    # =========================================================
    "IBM": {
        "2011-peak-revenue-but-rotting": {
            "year": 2011,
            "bucket": "mature_tech",
            "revenue": 106.92e9,
            "revenue_growth": 0.07,
            "gross_margin": 0.47,
            "op_margin": 0.20,
            "fcf_margin": 0.16,
            "net_margin": 0.15,
            "macro_context": "low-rates",
            "outcome": "loser",
            "notes": "Peak revenue ever. Stock $200 ATH. Buffett's biggest tech bet. Looked stable — but cloud disruption already cutting in.",
        },
        "2017-revenue-decline-trough": {
            "year": 2017,
            "bucket": "mature_tech",
            "revenue": 79.14e9,
            "revenue_growth": -0.01,
            "gross_margin": 0.46,
            "op_margin": 0.14,
            "fcf_margin": 0.16,
            "net_margin": 0.07,
            "macro_context": "peak-zirp",
            "outcome": "loser",
            "notes": "22 consecutive quarters of revenue decline (broken in Q4 2017). Revenue down $28B from 2011 peak. Buffett exited.",
        },
        "2020-red-hat-transition": {
            "year": 2020,
            "bucket": "mature_tech",
            "revenue": 73.62e9,
            "revenue_growth": -0.05,
            "gross_margin": 0.48,
            "op_margin": 0.10,
            "fcf_margin": 0.16,
            "net_margin": 0.075,
            "macro_context": "pandemic-zirp",
            "outcome": "loser",
            "notes": "Red Hat acquired (2019, $34B). Still flat/declining. Stock $120 — barely above 2010 levels. Decade of dead money.",
        },
        "2024-watsonx-ai-recovery": {
            "year": 2024,
            "bucket": "mature_tech",
            "revenue": 62.75e9,
            "revenue_growth": 0.015,
            "gross_margin": 0.57,
            "op_margin": 0.17,
            "fcf_margin": 0.20,
            "net_margin": 0.095,
            "macro_context": "ai-mania",
            "outcome": "winner",
            "notes": "watsonx AI book of business >$5B since launch. Software +9%. Stock $200 → $329 ATH early 2026. Genuine recovery, 13 years after 2011 peak.",
        },
    },

    # =========================================================
    # NOK — Nokia (smartphone era loser)
    # Arc: peak handset dominance → iPhone disruption → permanent collapse
    # CAUTION: pure disruption story — fundamentals great until they weren't
    # =========================================================
    "NOK": {
        "2007-peak-handset-dominance": {
            "year": 2007,
            "bucket": "mature_tech",
            "revenue": 51.0e9,  # EUR ~51B
            "revenue_growth": 0.24,
            "gross_margin": 0.33,
            "op_margin": 0.135,
            "fcf_margin": 0.13,
            "net_margin": 0.14,
            "macro_context": "pre-gfc",
            "outcome": "loser",
            "notes": "~50% global smartphone share. Stock $40 ATH. iPhone launched mid-2007 — Nokia management dismissed it. Classic incumbent denial.",
        },
        "2011-symbian-collapse": {
            "year": 2011,
            "bucket": "mature_tech",
            "revenue": 38.66e9,
            "revenue_growth": -0.07,
            "gross_margin": 0.29,
            "op_margin": -0.04,
            "fcf_margin": -0.02,
            "net_margin": -0.03,
            "macro_context": "post-gfc",
            "outcome": "loser",
            "notes": "Elop 'burning platform' memo. Bet on Windows Phone — major strategic mistake. Stock $4. Market share <10%.",
        },
        "2013-microsoft-sale": {
            "year": 2013,
            "bucket": "mature_tech",
            "revenue": 12.71e9,
            "revenue_growth": -0.18,
            "gross_margin": 0.29,
            "op_margin": -0.045,
            "fcf_margin": -0.05,
            "net_margin": -0.05,
            "macro_context": "post-gfc",
            "outcome": "loser",
            "notes": "Sold handset business to Microsoft for €5.4B. Total destruction of value: $250B market cap → $30B in 6 years.",
        },
    },

    # =========================================================
    # BBBY — Bed Bath & Beyond (retail disruption loser)
    # Arc: peak retail → Amazon disruption → bankruptcy
    # CAUTION: classic retail incumbent that couldn't transform
    # =========================================================
    "BBBY": {
        "2013-peak-revenue": {
            "year": 2013,
            "bucket": "consumer_cyclical",
            "revenue": 11.50e9,
            "revenue_growth": 0.06,
            "gross_margin": 0.395,
            "op_margin": 0.135,
            "fcf_margin": 0.09,
            "net_margin": 0.092,
            "macro_context": "low-rates",
            "outcome": "loser",
            "notes": "Near peak revenue. Stock $80. Coupon-driven category-killer. Amazon already eating share but few noticed.",
        },
        "2018-decline-emerging": {
            "year": 2018,
            "bucket": "consumer_cyclical",
            "revenue": 12.35e9,
            "revenue_growth": 0.022,
            "gross_margin": 0.34,
            "op_margin": 0.034,
            "fcf_margin": 0.04,
            "net_margin": 0.034,
            "macro_context": "peak-zirp",
            "outcome": "loser",
            "notes": "Top line peaked. Margins collapsing. Stock $12 (from $80). Activist Tritton brought in 2019.",
        },
        "2022-bankruptcy-spiral": {
            "year": 2022,
            "bucket": "consumer_cyclical",
            "revenue": 5.35e9,
            "revenue_growth": -0.40,
            "gross_margin": 0.27,
            "op_margin": -0.20,
            "fcf_margin": -0.30,
            "net_margin": -0.20,
            "macro_context": "rate-hikes",
            "outcome": "loser",
            "notes": "Meme stock chaos. Comparable sales -32%. Filed Chapter 11 April 2023. Total destruction. Lesson: retail moats are fragile.",
        },
    },

    # =========================================================
    # F — Ford Motor Company
    # Arc: GFC near-death → resilience → EV pivot struggle
    # MIXED: survived multiple crises but never sustained outperformance
    # =========================================================
    "F": {
        "2008-gfc-survival": {
            "year": 2008,
            "bucket": "consumer_cyclical",
            "revenue": 146.3e9,
            "revenue_growth": -0.15,
            "gross_margin": 0.05,
            "op_margin": -0.06,
            "fcf_margin": -0.10,
            "net_margin": -0.10,
            "macro_context": "gfc",
            "outcome": "mixed",
            "notes": "Lost $14.7B. Avoided bankruptcy (unlike GM/Chrysler). Mulally borrowed $23.6B pre-crisis. Stock $1-$2 lows.",
        },
        "2015-truck-renaissance": {
            "year": 2015,
            "bucket": "consumer_cyclical",
            "revenue": 149.6e9,
            "revenue_growth": 0.038,
            "gross_margin": 0.16,
            "op_margin": 0.06,
            "fcf_margin": 0.05,
            "net_margin": 0.05,
            "macro_context": "low-rates",
            "outcome": "mixed",
            "notes": "F-Series boom. Stock $15. Looked healthy but no real growth. Auto cycle peak — market saw it.",
        },
        "2022-ev-pivot-cash-burn": {
            "year": 2022,
            "bucket": "consumer_cyclical",
            "revenue": 158.06e9,
            "revenue_growth": 0.16,
            "gross_margin": 0.085,
            "op_margin": 0.043,
            "fcf_margin": 0.04,
            "net_margin": -0.013,
            "macro_context": "rate-hikes-ev-pivot",
            "outcome": "mixed",
            "notes": "Net loss despite revenue growth. Model E (EV) division losing $4B+/yr. F-150 Lightning unprofitable. Stock $11.",
        },
    },

    # =========================================================
    # DIS — Disney
    # Arc: peak parks/cable → streaming hope → streaming pain
    # MIXED: legacy assets still strong but transition painful
    # =========================================================
    "DIS": {
        "2018-peak-pre-streaming": {
            "year": 2018,
            "bucket": "consumer_cyclical",
            "revenue": 59.43e9,
            "revenue_growth": 0.078,
            "gross_margin": 0.45,
            "op_margin": 0.25,
            "fcf_margin": 0.16,
            "net_margin": 0.21,
            "macro_context": "peak-zirp",
            "outcome": "mixed",
            "notes": "Parks booming, cable still healthy, Fox acquisition announced. Stock $120. Pre-Disney+ profit machine.",
        },
        "2021-streaming-hope-peak": {
            "year": 2021,
            "bucket": "consumer_cyclical",
            "revenue": 67.42e9,
            "revenue_growth": 0.03,
            "gross_margin": 0.33,
            "op_margin": 0.06,
            "fcf_margin": 0.03,
            "net_margin": 0.03,
            "macro_context": "pandemic-zirp",
            "outcome": "mixed",
            "notes": "Stock $200 ATH. Disney+ at 116M subs. Streaming losing money but Wall Street believed in eventual profitability.",
        },
        "2023-iger-return-cost-cuts": {
            "year": 2023,
            "bucket": "consumer_cyclical",
            "revenue": 88.90e9,
            "revenue_growth": 0.072,
            "gross_margin": 0.33,
            "op_margin": 0.05,
            "fcf_margin": 0.05,
            "net_margin": 0.026,
            "macro_context": "rate-hikes",
            "outcome": "mixed",
            "notes": "Iger returned Nov 2022. $5.5B cost cuts, 7K layoffs. Streaming losses narrowing. Cord-cutting accelerating. Stock $80-100.",
        },
    },

    # =========================================================
    # BA — Boeing
    # Arc: peak pre-MAX → 737 MAX crisis → COVID compound disaster
    # CAUTION: classic safety/quality blowup — never fully recovered
    # =========================================================
    "BA": {
        "2018-peak-pre-max-crisis": {
            "year": 2018,
            "bucket": "industrial",
            "revenue": 101.13e9,
            "revenue_growth": 0.085,
            "gross_margin": 0.20,
            "op_margin": 0.118,
            "fcf_margin": 0.13,
            "net_margin": 0.10,
            "macro_context": "peak-zirp",
            "outcome": "loser",
            "notes": "Stock $440 ATH. Best year ever. THEN: Lion Air crash Oct 2018, Ethiopian crash Mar 2019. Entire 737 MAX fleet grounded.",
        },
        "2019-max-grounding": {
            "year": 2019,
            "bucket": "industrial",
            "revenue": 76.56e9,
            "revenue_growth": -0.24,
            "gross_margin": 0.17,
            "op_margin": -0.026,
            "fcf_margin": -0.03,
            "net_margin": -0.005,
            "macro_context": "late-zirp",
            "outcome": "loser",
            "notes": "MAX grounded all year. Revenue -24%. First annual loss since 1997. CEO Muilenburg fired. Stock -50%.",
        },
        "2020-covid-compound-disaster": {
            "year": 2020,
            "bucket": "industrial",
            "revenue": 58.16e9,
            "revenue_growth": -0.24,
            "gross_margin": -0.12,
            "op_margin": -0.21,
            "fcf_margin": -0.32,
            "net_margin": -0.20,
            "macro_context": "pandemic",
            "outcome": "loser",
            "notes": "MAX still grounded + COVID destroyed air travel. Massive losses. Took on $44B debt. Stock $89 low. Never sustained recovery to 2018 levels.",
        },
    },
# =========================================================
    # KO — Coca-Cola
    # Arc: sugar-tax fears stagnation → bottling refranchise margin lift → pricing power
    # =========================================================
    "KO": {
        "2015-sugar-tax-fears-stagnation": {
            "year": 2015,
            "bucket": "consumer_defensive",
            "revenue": 44.29e9,
            "revenue_growth": -0.037,
            "gross_margin": 0.605,
            "op_margin": 0.197,
            "fcf_margin": 0.18,
            "net_margin": 0.166,
            "macro_context": "low-rates-sugar-tax-era",
            "outcome": "winner",
            "notes": "Revenue declining 4-5% per year. Sugar tax narrative scared investors. Stock $40s, low-volume era. Pre-bottling-refranchise.",
        },
        "2018-bottling-refranchise-margin-lift": {
            "year": 2018,
            "bucket": "consumer_defensive",
            "revenue": 31.86e9,
            "revenue_growth": -0.099,  # divested bottling
            "gross_margin": 0.633,
            "op_margin": 0.269,
            "fcf_margin": 0.20,
            "net_margin": 0.20,
            "macro_context": "peak-zirp",
            "outcome": "winner",
            "notes": "Sold low-margin bottling operations. Revenue dropped but margins jumped. Higher-quality earnings.",
        },
        "2024-pricing-power-mature": {
            "year": 2024,
            "bucket": "consumer_defensive",
            "revenue": 47.06e9,
            "revenue_growth": 0.029,
            "gross_margin": 0.617,
            "op_margin": 0.318,
            "fcf_margin": 0.26,
            "net_margin": 0.226,
            "macro_context": "rate-hikes-inflation",
            "outcome": "winner",
            "notes": "Pricing power passed through inflation. Margins at decade-high. Slow steady compounder.",
        },
    },

    # =========================================================
    # WMT — Walmart
    # Arc: Amazon-fear era → ecommerce investments → omnichannel breakout
    # =========================================================
    "WMT": {
        "2016-amazon-fear-era": {
            "year": 2016,
            "bucket": "consumer_defensive",
            "revenue": 482.1e9,
            "revenue_growth": -0.007,
            "gross_margin": 0.252,
            "op_margin": 0.050,
            "fcf_margin": 0.03,
            "net_margin": 0.031,
            "macro_context": "low-rates-amazon-disruption-fear",
            "outcome": "winner",
            "notes": "Fiscal year ended Jan 2016. Revenue declining (FX + Amazon fear). Stock $60s. Lowest valuation in decade. Pre-Jet.com pivot.",
        },
        "2021-omnichannel-breakout": {
            "year": 2021,
            "bucket": "consumer_defensive",
            "revenue": 559.2e9,
            "revenue_growth": 0.067,
            "gross_margin": 0.243,
            "op_margin": 0.040,
            "fcf_margin": 0.05,
            "net_margin": 0.024,
            "macro_context": "pandemic-zirp",
            "outcome": "winner",
            "notes": "Fiscal year ended Jan 2021. eCommerce +79% in pandemic. Pickup/delivery infrastructure proven. Stock ATH.",
        },
        "2025-online-grocery-leader": {
            "year": 2025,
            "bucket": "consumer_defensive",
            "revenue": 680.99e9,
            "revenue_growth": 0.052,
            "gross_margin": 0.247,
            "op_margin": 0.042,
            "fcf_margin": 0.02,
            "net_margin": 0.030,
            "macro_context": "rate-hikes",
            "outcome": "winner",
            "notes": "Now #1 online grocery in US. Advertising business growing rapidly. Closed revenue gap to AMZN. Stock 3x'd from 2016 lows.",
        },
    },

    # =========================================================
    # COST — Costco
    # Arc: steady compounder phases — peak ZIRP, post-pandemic, pricing power
    # =========================================================
    "COST": {
        "2018-mature-compounder": {
            "year": 2018,
            "bucket": "consumer_defensive",
            "revenue": 138.43e9,
            "revenue_growth": 0.097,
            "gross_margin": 0.134,
            "op_margin": 0.032,
            "fcf_margin": 0.02,
            "net_margin": 0.023,
            "macro_context": "peak-zirp",
            "outcome": "winner",
            "notes": "Fiscal year ended Sep 2018. Membership fee growth steady. Stock $220. Best-in-class warehouse retailer.",
        },
        "2022-pandemic-pricing-resilience": {
            "year": 2022,
            "bucket": "consumer_defensive",
            "revenue": 226.95e9,
            "revenue_growth": 0.158,
            "gross_margin": 0.110,
            "op_margin": 0.034,
            "fcf_margin": 0.02,
            "net_margin": 0.026,
            "macro_context": "rate-hikes-inflation",
            "outcome": "winner",
            "notes": "Fiscal year ended Aug 2022. Pricing power + member loyalty intact during inflation. Stock $560.",
        },
        "2025-record-high": {
            "year": 2025,
            "bucket": "consumer_defensive",
            "revenue": 280.39e9,
            "revenue_growth": 0.082,
            "gross_margin": 0.111,
            "op_margin": 0.037,
            "fcf_margin": 0.04,
            "net_margin": 0.030,
            "macro_context": "rate-hikes",
            "outcome": "winner",
            "notes": "Membership fee at $5.3B. Stock ATH $1000+. Premium consumer staples valuation.",
        },
    },

    # =========================================================
    # PG — Procter & Gamble
    # Arc: stagnation/restructuring → pricing power compounder
    # =========================================================
    "PG": {
        "2015-stagnation-restructuring": {
            "year": 2015,
            "bucket": "consumer_defensive",
            "revenue": 76.28e9,
            "revenue_growth": -0.054,
            "gross_margin": 0.49,
            "op_margin": 0.181,
            "fcf_margin": 0.13,
            "net_margin": 0.10,
            "macro_context": "low-rates-strong-dollar",
            "outcome": "winner",
            "notes": "Currency headwinds. Divesting 100+ brands. Stock $80 sideways for years. Pre-Activist David Taylor era.",
        },
        "2020-pandemic-consumer-staples-bid": {
            "year": 2020,
            "bucket": "consumer_defensive",
            "revenue": 70.95e9,
            "revenue_growth": 0.049,
            "gross_margin": 0.499,
            "op_margin": 0.226,
            "fcf_margin": 0.20,
            "net_margin": 0.181,
            "macro_context": "pandemic-zirp",
            "outcome": "winner",
            "notes": "Bath/laundry brands ripped during pandemic. Margin expansion + organic growth returning.",
        },
        "2024-pricing-power-compounder": {
            "year": 2024,
            "bucket": "consumer_defensive",
            "revenue": 84.34e9,
            "revenue_growth": 0.024,
            "gross_margin": 0.510,
            "op_margin": 0.215,
            "fcf_margin": 0.18,
            "net_margin": 0.181,
            "macro_context": "rate-hikes",
            "outcome": "winner",
            "notes": "Passed through inflation cleanly. Margins recovering toward peak. Stock ATH ~$170.",
        },
    },

    # =========================================================
    # KHC — Kraft Heinz
    # Arc: post-merger margin peak → 3G playbook collapse → permanent re-rating
    # CAUTION: classic mistaken-margin-expansion story — 3G cost cuts looked great until brands eroded
    # =========================================================
    "KHC": {
        "2017-3g-playbook-peak": {
            "year": 2017,
            "bucket": "consumer_defensive",
            "revenue": 26.23e9,
            "revenue_growth": -0.003,
            "gross_margin": 0.378,
            "op_margin": 0.265,
            "fcf_margin": 0.20,
            "net_margin": 0.41,  # one-time tax reform benefit boosted net
            "macro_context": "peak-zirp",
            "outcome": "loser",
            "notes": "Margins looked stunning post-merger 3G cost cuts. Stock $90. Buffett/3G icons. WARNING: under-investment in brands becoming visible.",
        },
        "2019-3g-collapse": {
            "year": 2019,
            "bucket": "consumer_defensive",
            "revenue": 24.98e9,
            "revenue_growth": -0.048,
            "gross_margin": 0.336,
            "op_margin": -0.46,  # massive impairments
            "fcf_margin": 0.10,
            "net_margin": -0.20,
            "macro_context": "late-cycle",
            "outcome": "loser",
            "notes": "$15.4B writedown. Dividend cut 36%. SEC probe. Stock -50% to $25. Buffett admitted 'overpaid'. 3G playbook discredited.",
        },
        "2024-stagnation-stabilization": {
            "year": 2024,
            "bucket": "consumer_defensive",
            "revenue": 25.85e9,
            "revenue_growth": -0.024,
            "gross_margin": 0.347,
            "op_margin": 0.065,
            "fcf_margin": 0.14,
            "net_margin": 0.106,
            "macro_context": "rate-hikes",
            "outcome": "loser",
            "notes": "Revenue flat $25B for 8+ years. Stock $30 — half of 2017 peak. Demerger announced. Never recovered.",
        },
    },

}


def all_anchor_points():
    """Flatten ANCHORS into a list of (ticker, phase_label, data_dict) tuples."""
    points = []
    for ticker, phases in ANCHORS.items():
        for phase_label, data in phases.items():
            points.append((ticker, phase_label, data))
    return points


if __name__ == "__main__":
    # Sanity check: print summary
    total = 0
    for ticker, phases in ANCHORS.items():
        print(f"{ticker}: {len(phases)} phases")
        for phase_label, data in phases.items():
            print(f"  {data['year']} | {phase_label:35s} | bucket={data['bucket']:18s} | rev=${data['revenue']/1e9:.1f}B | growth={data['revenue_growth']*100:+.1f}% | op_margin={data['op_margin']*100:+.1f}%")
        total += len(phases)
    print(f"\nTotal anchor points so far: {total}")

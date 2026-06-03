import sqlite3
import os

DB_PATH = "data.db"


def init_db():
    """Create database tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS snapshots (
            ticker TEXT,
            date TEXT,
            price REAL,
            rsi REAL,
            ma_20 REAL,
            ma_50 REAL,
            ma_200 REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        );

        CREATE TABLE IF NOT EXISTS fundamentals (
            ticker TEXT PRIMARY KEY,
            updated_at TEXT,
            sector TEXT,
            pe REAL,
            op_margin REAL,
            roe REAL,
            revenue_growth REAL,
            earnings_growth REAL,
            debt_equity REAL,
            free_cash_flow_positive INTEGER,
            institutional_ownership REAL,
            insider_ownership REAL,
            fund_score REAL,
            max_score REAL,
            core_passed INTEGER
        );

        CREATE TABLE IF NOT EXISTS sector_benchmarks (
            sector TEXT PRIMARY KEY,
            pe_median REAL,
            op_margin_median REAL,
            roe_median REAL,
            updated_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_snapshots_date ON snapshots(date);
        CREATE INDEX IF NOT EXISTS idx_snapshots_ticker ON snapshots(ticker);
    """)
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DB_PATH}")


def get_latest_snapshot(ticker):
    """Get the most recent snapshot for a single ticker."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("""
        SELECT ticker, date, price, rsi, ma_20, ma_50, ma_200, volume
        FROM snapshots
        WHERE ticker = ?
        ORDER BY date DESC LIMIT 1
    """, (ticker,)).fetchone()
    conn.close()

    if not row:
        return None

    return {
        'ticker': row[0],
        'date': row[1],
        'price': row[2],
        'rsi': row[3],
        'ma_20': row[4],
        'ma_50': row[5],
        'ma_200': row[6],
        'volume': row[7],
    }


def get_all_latest_snapshots():
    """Get the latest snapshot for every ticker in the database."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT s.ticker, s.date, s.price, s.rsi, s.ma_20, s.ma_50, s.ma_200, s.volume
        FROM snapshots s
        INNER JOIN (
            SELECT ticker, MAX(date) AS max_date
            FROM snapshots
            GROUP BY ticker
        ) latest ON s.ticker = latest.ticker AND s.date = latest.max_date
    """).fetchall()
    conn.close()

    return [{
        'ticker': r[0], 'date': r[1], 'price': r[2], 'rsi': r[3],
        'ma_20': r[4], 'ma_50': r[5], 'ma_200': r[6], 'volume': r[7]
    } for r in rows]


def get_historical_prices(ticker, days=30):
    """Get last N days of price history for a ticker."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT date, price FROM snapshots
        WHERE ticker = ?
        ORDER BY date DESC LIMIT ?
    """, (ticker, days)).fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    init_db()

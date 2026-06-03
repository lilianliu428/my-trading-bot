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


if __name__ == "__main__":
    init_db()
    
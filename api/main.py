"""
api/main.py

FastAPI application entry point. This file creates the FastAPI app and
registers all routers.

Run with:
    uvicorn api.main:app --reload --port 8000

Then visit:
    http://localhost:8000/                    sanity check
    http://localhost:8000/docs                interactive API documentation
    http://localhost:8000/api/valuation/NVDA  example valuation
"""

from fastapi import FastAPI

from api.routers import valuation

app = FastAPI(
    title="XPocketTrader API",
    description="Valuation engine backend for the XPocketTrader bot and dashboard.",
    version="0.1.0",
)

# Register routers — each one adds its endpoints to the app
app.include_router(valuation.router)


@app.get("/")
def root():
    """Sanity-check endpoint. Returns a hello message so we know the API is up."""
    return {"message": "XPocketTrader API is running", "version": "0.1.0"}
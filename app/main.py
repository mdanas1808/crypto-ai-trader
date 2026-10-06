from fastapi import FastAPI

from app.api.routes import router
from app.config.settings import settings

app = FastAPI(title="Crypto AI Trader")
app.include_router(router)

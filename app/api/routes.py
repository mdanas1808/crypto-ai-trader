from fastapi import APIRouter

from app.config.settings import settings

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "Crypto AI Trader",
        "environment": settings.app_env,
    }

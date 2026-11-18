"""
api/routes package
"""
from api.routes.health import router as health_router
from api.routes.model_info import router as model_info_router
from api.routes.map_match import router as map_match_router

__all__ = ["health_router", "model_info_router", "map_match_router"]

"""FastAPI application for CrisisWorld server."""

from openenv.core.env_server import create_fastapi_app

from models import ActionUnion, Observation
from server.CrisisWorld_environment import CrisisWorld

env = CrisisWorld()
app = create_fastapi_app(env, ActionUnion, Observation)

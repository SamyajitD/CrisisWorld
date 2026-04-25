"""FastAPI entrypoint via OpenEnv create_app().

Wires CrisisWorld, OuterAction, and Observation
into the OpenEnv HTTP server per CLAUDE.md §5.2.
"""

from __future__ import annotations


# from openenv.core.env_server import create_web_interface_app as create_app # For Web interface
from openenv.core.env_server.http_server import create_app
from ..models import OuterAction, Observation

# from ..models import TriageSieveAction, TriageSieveObservation
from ..server.CrisisWorld_environment import CrisisWorld

app = create_app(
    env=CrisisWorld,
    action_cls=OuterAction,
    observation_cls=Observation,
    env_name="crisis_world",
    max_concurrent_envs=1,
)

__all__ = ["app", "main"]

def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="CrisisWorld-OpenEnv server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
    

if __name__ == "__main__":
    main()
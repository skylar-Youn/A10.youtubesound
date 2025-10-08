"""Root-level ASGI entrypoint for `uvicorn server:app`.

Imports the FastAPI application from `prosodynet_project.server`.
"""

from prosodynet_project.server import app  # noqa: F401


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)

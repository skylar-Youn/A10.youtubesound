"""Root-level ASGI entrypoint for `uvicorn server:app`.

Imports the FastAPI application from `prosodynet_project.server`.
"""

from prosodynet_project.server import app  # noqa: F401


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7000, reload=False)

from setuptools import setup, find_packages

setup(
    name="prosodynet_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "librosa",
        "pyworld",
        "soundfile",
        "fastapi",
        "uvicorn",
        "pydantic",
    ],
)

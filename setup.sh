#!/bin/bash

uv sync
uv run pre-commit install

uv pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==25.8.*

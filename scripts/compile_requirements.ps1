$ErrorActionPreference = "Stop"

python -m pip install -U pip pip-tools
pip-compile requirements.in -o requirements.txt

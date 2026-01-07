# pokepocketsim (Windows / Python 3.10)

## Dependency install (Windows)

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Regenerate lock files (pip-tools)

Edit only the `.in` files and regenerate the locks via pip-compile.
Do **not** hand-edit `requirements.txt`.

```powershell
. .venv\Scripts\Activate.ps1
scripts\compile_requirements.ps1
```

### Torch CPU/GPU policy

* Default is CPU-only torch using `requirements-torch-cpu.in` (uses the PyTorch CPU index).
* To switch to GPU, create a new torch input file and lock it separately, then recompile the main lock using that constraint.

Example (CUDA 12.1; adjust to your target):

```powershell
@"
--extra-index-url https://download.pytorch.org/whl/cu121

torch
"@ | Set-Content requirements-torch-gpu.in

pip-compile requirements-torch-gpu.in -o requirements-torch-gpu.txt
pip-compile requirements.in -o requirements.txt -c requirements-torch-gpu.txt
```

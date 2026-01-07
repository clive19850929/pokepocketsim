# Operations Notes

## Dependency management (pip-tools)

1. Install/update pip-tools:
   ```powershell
   python -m pip install -U pip pip-tools
   ```
2. Update pinned requirements:
   ```powershell
   pip-compile requirements.in -o requirements.txt
   ```
   Or run the helper script:
   ```powershell
   scripts\\compile_requirements.ps1
   ```
3. Install from pinned requirements:
   ```powershell
   python -m pip install -r requirements.txt
   ```

## Diff + log collection (Cursor)

After changes, save the diff so it can be attached with Cursor logs/debug dumps:

```powershell
git diff > fixes\\issue_YYYYMMDD_short_desc.diff
```

Attach `fixes\\*.diff` together with `run_*.log` and any `debug_dump_*.json` files when reporting issues.

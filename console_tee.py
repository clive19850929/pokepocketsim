#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import atexit

# --- console tee (stdout/stderr -> single game_id.log) ---
_TEE_FP = None
_TEE_STDOUT0 = None
_TEE_STDERR0 = None


class _TeeStream:
    def __init__(self, base, fp):
        self._base = base
        self._fp = fp

        self._console_tee_active = True
        try:
            self._console_tee_path = os.path.abspath(getattr(fp, "name", "") or "")
        except Exception:
            self._console_tee_path = ""

    def write(self, s):
        try:
            self._base.write(s)
        except Exception:
            pass
        try:
            self._fp.write(s)
        except Exception:
            pass
        try:
            return len(s)
        except Exception:
            return 0

    def flush(self):
        try:
            self._base.flush()
        except Exception:
            pass
        try:
            self._fp.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return bool(self._base.isatty())
        except Exception:
            return False

    @property
    def encoding(self):
        try:
            return getattr(self._base, "encoding", "utf-8")
        except Exception:
            return "utf-8"


def _console_tee_active() -> bool:
    try:
        obj = getattr(sys, "stdout", None)
        for _ in range(16):
            if obj is None:
                break
            if getattr(obj, "_console_tee_active", False):
                return True
            obj = getattr(obj, "_base", None)
    except Exception:
        pass
    return False


def _console_tee_fp_and_path():
    try:
        obj = getattr(sys, "stdout", None)
        for _ in range(16):
            if obj is None:
                break
            if getattr(obj, "_console_tee_active", False):
                fp = getattr(obj, "_fp", None)
                if fp is not None:
                    try:
                        return fp, os.path.abspath(getattr(fp, "name", "") or "")
                    except Exception:
                        return fp, ""
            obj = getattr(obj, "_base", None)
    except Exception:
        pass
    return None, ""


def _tee_console_start(log_path: str, enable: bool = True) -> None:
    # 旧系統(_TeeStream)は使わず、新系統(_ConsoleTeeStream)に統一する
    try:
        _setup_console_tee_to_file(log_path, enable=enable)
    except Exception:
        pass

def _tee_console_stop() -> None:
    # _setup_console_tee 側の close を呼ぶ（登録済み atexit と同等）
    global _CONSOLE_TEE_FP, _CONSOLE_TEE_STDOUT0, _CONSOLE_TEE_STDERR0, _CONSOLE_TEE_OWNED

    if not bool(_CONSOLE_TEE_OWNED):
        _CONSOLE_TEE_FP = None
        _CONSOLE_TEE_STDOUT0 = None
        _CONSOLE_TEE_STDERR0 = None
        _CONSOLE_TEE_OWNED = False
        return

    try:
        if _CONSOLE_TEE_STDOUT0 is not None:
            sys.stdout = _CONSOLE_TEE_STDOUT0
        if _CONSOLE_TEE_STDERR0 is not None:
            sys.stderr = _CONSOLE_TEE_STDERR0
    except Exception:
        pass
    try:
        if _CONSOLE_TEE_FP is not None:
            _CONSOLE_TEE_FP.flush()
            _CONSOLE_TEE_FP.close()
    except Exception:
        pass
    _CONSOLE_TEE_FP = None
    _CONSOLE_TEE_STDOUT0 = None
    _CONSOLE_TEE_STDERR0 = None
    _CONSOLE_TEE_OWNED = False

class _ConsoleTeeStream:
    def __init__(self, base, fp):
        self._base = base
        self._fp = fp

        self._console_tee_active = True
        try:
            self._console_tee_path = os.path.abspath(getattr(fp, "name", "") or "")
        except Exception:
            self._console_tee_path = ""

    def write(self, s):
        try:
            self._base.write(s)
        except Exception:
            pass
        try:
            self._fp.write(s)
        except Exception:
            pass
        try:
            return len(s)
        except Exception:
            return 0

    def flush(self):
        try:
            self._base.flush()
        except Exception:
            pass
        try:
            self._fp.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return bool(self._base.isatty())
        except Exception:
            return False

    @property
    def encoding(self):
        try:
            return getattr(self._base, "encoding", "utf-8")
        except Exception:
            return "utf-8"

def _setup_console_tee(log_dir, prefix="ai_vs_ai_console", enable=True, fixed_path=None):
    global _CONSOLE_TEE_FP, _CONSOLE_TEE_STDOUT0, _CONSOLE_TEE_STDERR0, _CONSOLE_TEE_OWNED

    if not enable:
        return
    if _CONSOLE_TEE_FP is not None:
        return

    # _base チェーンに Tee が居たら二重にラップしない（nested tee を確実に防ぐ）
    if _console_tee_active():
        try:
            if fixed_path:
                fp0, fp0_path = _console_tee_fp_and_path()
                if fp0 is not None and fp0_path:
                    target = os.path.normcase(os.path.normpath(os.path.abspath(str(fixed_path))))
                    cur = os.path.normcase(os.path.normpath(os.path.abspath(str(fp0_path))))
                    if cur == target:
                        _CONSOLE_TEE_FP = fp0
                        _CONSOLE_TEE_OWNED = False
                        return
        except Exception:
            pass
        return

    # 既に stdout が Tee 済みなら二重にラップしない（nested tee を防いで二重化/空行を防止）
    try:
        so = getattr(sys, "stdout", None)
        if so is not None and getattr(so, "__class__", None) is not None:
            if so.__class__.__name__ in ("_TeeStream", "_ConsoleTeeStream"):
                # fixed_path が同一なら「既に目的の tee」とみなして終了
                try:
                    if fixed_path:
                        target = os.path.normcase(os.path.normpath(os.path.abspath(str(fixed_path))))
                        fp0 = getattr(so, "_fp", None)
                        fp0_name = getattr(fp0, "name", None) if fp0 is not None else None
                        if fp0_name:
                            cur = os.path.normcase(os.path.normpath(os.path.abspath(str(fp0_name))))
                            if cur == target:
                                _CONSOLE_TEE_FP = fp0
                                _CONSOLE_TEE_OWNED = False
                                return
                except Exception:
                    pass
                # それ以外でも nested tee は避ける
                return
    except Exception:
        pass


    try:
        if fixed_path:
            path = fixed_path
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
        else:
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, f"{prefix}_{os.getpid()}.log")
    except Exception:
        return

    try:
        fp = open(path, "a", encoding="utf-8", buffering=1)
    except Exception:
        return

    try:
        _CONSOLE_TEE_STDOUT0 = sys.stdout
        _CONSOLE_TEE_STDERR0 = sys.stderr
        sys.stdout = _ConsoleTeeStream(_CONSOLE_TEE_STDOUT0, fp)
        sys.stderr = _ConsoleTeeStream(_CONSOLE_TEE_STDERR0, fp)
        _CONSOLE_TEE_FP = fp
        _CONSOLE_TEE_OWNED = True

        def _close():
            global _CONSOLE_TEE_FP, _CONSOLE_TEE_STDOUT0, _CONSOLE_TEE_STDERR0, _CONSOLE_TEE_OWNED

            if not bool(_CONSOLE_TEE_OWNED):
                _CONSOLE_TEE_FP = None
                _CONSOLE_TEE_STDOUT0 = None
                _CONSOLE_TEE_STDERR0 = None
                _CONSOLE_TEE_OWNED = False
                return

            try:
                if _CONSOLE_TEE_STDOUT0 is not None:
                    sys.stdout = _CONSOLE_TEE_STDOUT0
                if _CONSOLE_TEE_STDERR0 is not None:
                    sys.stderr = _CONSOLE_TEE_STDERR0
            except Exception:
                pass
            try:
                if _CONSOLE_TEE_FP is not None:
                    _CONSOLE_TEE_FP.flush()
                    _CONSOLE_TEE_FP.close()
            except Exception:
                pass
            _CONSOLE_TEE_FP = None
            _CONSOLE_TEE_STDOUT0 = None
            _CONSOLE_TEE_STDERR0 = None
            _CONSOLE_TEE_OWNED = False

        atexit.register(_close)
    except Exception:
        try:
            fp.close()
        except Exception:
            pass
        return

def _setup_console_tee_to_file(log_path: str, enable: bool = True) -> None:
    try:
        d = os.path.dirname(log_path)
        if not d:
            d = "."
    except Exception:
        d = "."
    _setup_console_tee(d, prefix="ai_vs_ai_console", enable=enable, fixed_path=log_path)

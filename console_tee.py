#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import atexit

# --- console tee (stdout/stderr -> single game_id.log) ---
_TEE_FP = None
_TEE_STDOUT0 = None
_TEE_STDERR0 = None

_CONSOLE_TEE_FP = None
_CONSOLE_TEE_STDOUT0 = None
_CONSOLE_TEE_STDERR0 = None
_CONSOLE_TEE_STDERR_CONSOLE = None
_CONSOLE_TEE_OWNED = False

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


def _console_tee_matches_path(target_path: str) -> bool:
    """
    現在 stdout が console_tee 済みで、その出力先が target_path と一致するか。

    - sys.stdout の _base チェーンを辿る（_console_tee_fp_and_path を利用）
    - パスは normcase/normpath/abspath で正規化して比較
    """
    try:
        fp0, fp0_path = _console_tee_fp_and_path()
        if fp0 is None:
            return False
        if not fp0_path:
            return False

        target = os.path.normcase(os.path.normpath(os.path.abspath(str(target_path))))
        cur = os.path.normcase(os.path.normpath(os.path.abspath(str(fp0_path))))
        return cur == target
    except Exception:
        return False

def _tee_console_start(log_path: str, enable: bool = True) -> None:
    # 旧系統(_TeeStream)は使わず、新系統(_ConsoleTeeStream)に統一する
    try:
        _setup_console_tee_to_file(log_path, enable=enable)
    except Exception:
        pass

def _tee_console_stop() -> None:
    # _setup_console_tee 側の close を呼ぶ（登録済み atexit と同等）
    global _CONSOLE_TEE_FP, _CONSOLE_TEE_STDOUT0, _CONSOLE_TEE_STDERR0, _CONSOLE_TEE_OWNED, _CONSOLE_TEE_STDERR_CONSOLE

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

    # restore OS-level stderr (fd=2) BEFORE closing fp
    try:
        if _CONSOLE_TEE_STDERR_CONSOLE is not None:
            try:
                os.dup2(_CONSOLE_TEE_STDERR_CONSOLE.fileno(), 2)
            except Exception:
                pass
            try:
                _CONSOLE_TEE_STDERR_CONSOLE.close()
            except Exception:
                pass
            _CONSOLE_TEE_STDERR_CONSOLE = None
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

        self._console_tee_hide = []
        try:
            v = os.getenv("CONSOLE_TEE_HIDE_CONSOLE_SUBSTRINGS", "") or ""
            for x in v.split(","):
                x = x.strip()
                if x:
                    self._console_tee_hide.append(x)
        except Exception:
            pass

        self._console_tee_linebuf = ""

    def _should_drop_console_line(self, line: str, hide) -> bool:
        try:
            if os.getenv("AZ_DECISION_LOG_FILE_ONLY", "0") == "1":
                if "[AZ][DECISION][CALL]" in line:
                    return True
        except Exception:
            pass

        try:
            if os.getenv("AZ_DECISION_HIDE_CALL_CONSOLE", "0") == "1":
                if "[AZ][DECISION][CALL]" in line:
                    return True
        except Exception:
            pass

        if hide:
            for pat in hide:
                try:
                    if pat and pat in line:
                        return True
                except Exception:
                    pass

        return False

    def write(self, s):
        try:
            if not isinstance(s, str):
                s = str(s)
        except Exception:
            s = ""

        try:
            self._fp.write(s)
        except Exception:
            pass

        hide = None
        try:
            hide = getattr(self, "_console_tee_hide", None)
        except Exception:
            hide = None

        try:
            buf = getattr(self, "_console_tee_linebuf", "")
            if not isinstance(buf, str):
                buf = str(buf)
            buf = buf + s

            while True:
                idx = buf.find("\n")
                if idx < 0:
                    break

                line = buf[: idx + 1]
                buf = buf[idx + 1 :]

                drop = False
                try:
                    drop = self._should_drop_console_line(line, hide)
                except Exception:
                    drop = False

                if not drop:
                    try:
                        self._base.write(line)
                    except Exception:
                        pass

            self._console_tee_linebuf = buf
        except Exception:
            try:
                self._base.write(s)
            except Exception:
                pass

        try:
            return len(s)
        except Exception:
            return 0

    def flush(self):
        hide = None
        try:
            hide = getattr(self, "_console_tee_hide", None)
        except Exception:
            hide = None

        try:
            buf = getattr(self, "_console_tee_linebuf", "")
            if buf:
                drop = False
                try:
                    drop = self._should_drop_console_line(buf, hide)
                except Exception:
                    drop = False
                if not drop:
                    try:
                        self._base.write(buf)
                    except Exception:
                        pass
                self._console_tee_linebuf = ""
        except Exception:
            pass

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

        # NOTE: os.write(2, ...) 等で OS レベル fd=2 に直接書かれる場合があるため、
        #       sys.stderr 差し替えだけでは拾えない。fd=2 もログへ向ける。
        try:
            global _CONSOLE_TEE_STDERR_CONSOLE
            _fd2 = os.dup(2)
            os.dup2(fp.fileno(), 2)
            _CONSOLE_TEE_STDERR_CONSOLE = os.fdopen(_fd2, "w", encoding="utf-8", buffering=1)
        except Exception:
            _CONSOLE_TEE_STDERR_CONSOLE = None

        sys.stdout = _ConsoleTeeStream(_CONSOLE_TEE_STDOUT0, fp)
        _stderr0 = _CONSOLE_TEE_STDERR_CONSOLE if _CONSOLE_TEE_STDERR_CONSOLE is not None else _CONSOLE_TEE_STDERR0
        sys.stderr = _ConsoleTeeStream(_stderr0, fp)
        _CONSOLE_TEE_FP = fp
        _CONSOLE_TEE_OWNED = True

        def _close():
            global _CONSOLE_TEE_FP, _CONSOLE_TEE_STDOUT0, _CONSOLE_TEE_STDERR0, _CONSOLE_TEE_OWNED, _CONSOLE_TEE_STDERR_CONSOLE

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

            # restore OS-level stderr (fd=2) BEFORE closing fp
            try:
                if _CONSOLE_TEE_STDERR_CONSOLE is not None:
                    try:
                        os.dup2(_CONSOLE_TEE_STDERR_CONSOLE.fileno(), 2)
                    except Exception:
                        pass
                    try:
                        _CONSOLE_TEE_STDERR_CONSOLE.close()
                    except Exception:
                        pass
                    _CONSOLE_TEE_STDERR_CONSOLE = None
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

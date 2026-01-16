#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import re
import sys
import threading
from datetime import datetime


def _safe_filename(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    if not s:
        s = "unknown_game"
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = re.sub(r"\s+", "_", s)
    return s[:200]


class TeeTextIO(io.TextIOBase):
    def __init__(self, *streams, autoflush: bool = True):
        self._streams = streams
        self._autoflush = autoflush
        self._lock = threading.Lock()

    def write(self, s):
        if s is None:
            return 0
        with self._lock:
            for st in self._streams:
                try:
                    st.write(s)
                except Exception:
                    pass
                if self._autoflush:
                    try:
                        st.flush()
                    except Exception:
                        pass
        return len(s)

    def flush(self):
        with self._lock:
            for st in self._streams:
                try:
                    st.flush()
                except Exception:
                    pass


class GameLogContext:
    """
    1ゲーム単位で stdout/stderr をファイルへ tee するコンテキスト。
    - ファイル名: {game_id}.log
    - ディレクトリ: log_dir
    """
    def __init__(self, game_id: str, log_dir: str = "game_logs", to_console: bool = True, encoding: str = "utf-8"):
        self._game_id = game_id
        self._log_dir = log_dir
        self._to_console = bool(to_console)
        self._encoding = encoding

        self._path = None
        self._fp = None
        self._orig_out = None
        self._orig_err = None

    @property
    def path(self) -> str:
        return self._path

    def __enter__(self):
        os.makedirs(self._log_dir, exist_ok=True)

        fname = _safe_filename(self._game_id) + ".log"
        path = os.path.join(self._log_dir, fname)
        self._path = path

        self._fp = None

        # 既に console_tee が同一パスへ tee 済みなら、ここでは stdout/stderr を差し替えない
        try:
            _matches = None
            try:
                from console_tee import _console_tee_matches_path as _matches
            except Exception:
                _matches = None

            if _matches is not None:
                try:
                    if _matches(self._path):
                        self._orig_out = sys.stdout
                        self._orig_err = sys.stderr

                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[LOG] ===== game log begin ===== ts={ts} game_id={self._game_id}")
                        print(f"[LOG] path={os.path.abspath(self._path)}")
                        print(f"[LOG] pid={os.getpid()} python={sys.version.split()[0]}")
                        print(f"[LOG] cwd={os.getcwd()}")
                        try:
                            print(f"[LOG] argv={sys.argv}")
                        except Exception:
                            pass
                        print(f"[LOG] ===========================")
                        return self
                except Exception:
                    pass
        except Exception:
            pass

        # 互換: console_tee.py が無い/読めない場合は従来ロジックで判定する
        try:
            target = os.path.normcase(os.path.normpath(os.path.abspath(str(self._path))))
            obj = getattr(sys, "stdout", None)
            for _ in range(16):
                if obj is None:
                    break
                if getattr(obj, "_console_tee_active", False):
                    cur = getattr(obj, "_console_tee_path", "") or ""
                    if not cur:
                        fp0 = getattr(obj, "_fp", None)
                        cur = getattr(fp0, "name", "") if fp0 is not None else ""
                    cur = os.path.normcase(os.path.normpath(os.path.abspath(str(cur))))
                    if cur == target:
                        self._orig_out = sys.stdout
                        self._orig_err = sys.stderr

                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[LOG] ===== game log begin ===== ts={ts} game_id={self._game_id}")
                        print(f"[LOG] path={os.path.abspath(self._path)}")
                        print(f"[LOG] pid={os.getpid()} python={sys.version.split()[0]}")
                        print(f"[LOG] cwd={os.getcwd()}")
                        try:
                            print(f"[LOG] argv={sys.argv}")
                        except Exception:
                            pass
                        print(f"[LOG] ===========================")
                        return self
                obj = getattr(obj, "_base", None)
        except Exception:
            pass

        self._fp = open(self._path, "a", encoding=self._encoding, buffering=1)

        self._orig_out = sys.stdout
        self._orig_err = sys.stderr

        if self._to_console:
            _base_out = self._orig_out
            _base_err = self._orig_err
            sys.stdout = TeeTextIO(_base_out, self._fp, autoflush=True)
            sys.stderr = TeeTextIO(_base_err, self._fp, autoflush=True)
        else:
            sys.stdout = TeeTextIO(self._fp, autoflush=True)
            sys.stderr = TeeTextIO(self._fp, autoflush=True)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[LOG] ===== game log begin ===== ts={ts} game_id={self._game_id}")
        print(f"[LOG] path={os.path.abspath(self._path)}")
        print(f"[LOG] pid={os.getpid()} python={sys.version.split()[0]}")
        print(f"[LOG] cwd={os.getcwd()}")
        try:
            print(f"[LOG] argv={sys.argv}")
        except Exception:
            pass
        print(f"[LOG] ===========================")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if exc_type is None:
                print(f"[LOG] ===== game log end ===== ts={ts} status=OK")
            else:
                print(f"[LOG] ===== game log end ===== ts={ts} status=EXC type={exc_type.__name__} msg={exc}")
        except Exception:
            pass

        try:
            sys.stdout = self._orig_out if self._orig_out is not None else sys.__stdout__
            sys.stderr = self._orig_err if self._orig_err is not None else sys.__stderr__
        except Exception:
            pass

        try:
            if self._fp is not None:
                self._fp.flush()
        except Exception:
            pass

        try:
            if self._fp is not None:
                self._fp.close()
        except Exception:
            pass

        self._fp = None
        return False

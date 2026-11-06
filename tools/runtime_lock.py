"""Cross-process application lease shared with the Windows installer/updater."""
from __future__ import annotations

import atexit
import ctypes
from ctypes import wintypes
import hashlib
import os
from pathlib import Path
import threading

_LOCAL_LOCK = threading.RLock()
_HELD = set()
_GUI_LEASE = None


def lock_name(root: Path, kind: str = 'app') -> str:
    canonical = str(Path(root).resolve()).replace('/', '\\').rstrip('\\').lower()
    digest = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    return f'Local\\IndexTTSEasyGUI-{kind}-{digest}'


def require_clean_update(root: Path) -> None:
    if (Path(root) / '.runtime/update/transaction.json').exists():
        raise RuntimeError('An interrupted update needs recovery. Close this window and run 3- update.bat first.')


def gui_application_lock(root: Path):
    """Keep one same-process lease through launcher preflight and GUI execution."""
    global _GUI_LEASE
    with _LOCAL_LOCK:
        if _GUI_LEASE is None:
            _GUI_LEASE = ApplicationLock(root).__enter__()
        elif _GUI_LEASE.name != lock_name(root):
            raise RuntimeError('A different GUI deployment is already active in this process.')
        return _GUI_LEASE


class ApplicationLock:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.name = lock_name(self.root)
        self.handle = None
        self._fallback = None

    def __enter__(self):
        with _LOCAL_LOCK:
            if self.name in _HELD:
                raise RuntimeError('The GUI, installer or updater is already running for this directory.')
            if os.name == 'nt':
                kernel = ctypes.WinDLL('kernel32', use_last_error=True)
                kernel.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
                kernel.CreateMutexW.restype = wintypes.HANDLE
                kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
                kernel.WaitForSingleObject.restype = wintypes.DWORD
                kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
                kernel.ReleaseMutex.restype = wintypes.BOOL
                kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                kernel.CloseHandle.restype = wintypes.BOOL
                handle = kernel.CreateMutexW(None, False, self.name)
                if not handle:
                    raise ctypes.WinError(ctypes.get_last_error())
                wait = kernel.WaitForSingleObject(handle, 0)
                if wait not in (0, 0x80):  # Acquired, or abandoned by an exited process.
                    kernel.CloseHandle(handle)
                    raise RuntimeError('The GUI, installer or updater is already running for this directory.')
                self._kernel = kernel
                self.handle = handle
            else:
                import fcntl
                folder = self.root / '.runtime/update'
                folder.mkdir(parents=True, exist_ok=True)
                stream = (folder / 'application.lock').open('a+b')
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    stream.close()
                    raise RuntimeError('The GUI, installer or updater is already running for this directory.')
                self._fallback = stream
            _HELD.add(self.name)
            atexit.register(self.close)
            return self

    def close(self):
        with _LOCAL_LOCK:
            if self.handle is not None:
                self._kernel.ReleaseMutex(self.handle)
                self._kernel.CloseHandle(self.handle)
                self.handle = None
                _HELD.discard(self.name)
            if self._fallback is not None:
                self._fallback.close()
                self._fallback = None
                _HELD.discard(self.name)

    def __exit__(self, *args):
        self.close()

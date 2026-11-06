"""Hold the application lease from environment preflight through GUI shutdown."""
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from tools.runtime_lock import gui_application_lock, require_clean_update
    gui_application_lock(ROOT)
    require_clean_update(ROOT)
    for script in ('ensure_ffmpeg.py', 'verify_cuda.py', 'verify_numpy.py'):
        result = subprocess.run([sys.executable, '-B', str(ROOT/'tools'/script)], cwd=ROOT)
        if result.returncode:
            raise SystemExit(result.returncode)
    runpy.run_path(str(ROOT/'easy_gui.py'), run_name='__main__')


if __name__ == '__main__':
    main()

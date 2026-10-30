from pathlib import Path
import shutil
import subprocess

import imageio_ffmpeg

root = Path(__file__).resolve().parents[1]
ffmpeg = root / ".runtime" / "bin" / "ffmpeg.exe"
ffmpeg.parent.mkdir(parents=True, exist_ok=True)
if not ffmpeg.is_file():
    shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg)
subprocess.run([str(ffmpeg), "-version"], check=True, stdout=subprocess.DEVNULL)
print(f"FFmpeg OK: {ffmpeg}")
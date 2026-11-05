from __future__ import annotations
import json, threading, time, subprocess, sys, webbrowser
from pathlib import Path

def _play_done_sound():
    if __import__('os').name != 'nt': return
    try:
        import winsound
        root=Path(__file__).resolve().parents[1]
        candidates=[root/'assets'/'inference_training_done.wav', root/'assets'/'chime.wav', root/'assets'/'inference_complete.wav']
        wav=next((x for x in candidates if x.is_file()),None)
        if wav: winsound.PlaySound(str(wav),winsound.SND_FILENAME|winsound.SND_ASYNC)
    except Exception: pass

class TrainingManager:
    def __init__(self):
        self.lock=threading.RLock(); self.thread=None; self.stop_event=threading.Event(); self.tensorboard=None; self.tensorboard_logdir=None
        self.state={'running':False,'step':0,'total':0,'epoch':0,'loss':None,'elapsed':0.0,'eta':0.0,'message':'Idle.','output':''}
    def _update(self,**kw):
        with self.lock: self.state.update({k:v for k,v in kw.items() if v is not None})
    def snapshot(self):
        with self.lock: return dict(self.state)
    def start(self,config):
        from .dataset import DATASET_LOCK, load_project, validate_project_manifest
        from .proofread import directory_lock
        from contextlib import nullcontext
        with DATASET_LOCK:
            training_root = config.get('training_root', 'training')
            project = load_project(config.get('project'), training_root)
            source = project.get('source_folder')
            with directory_lock(source) if source else nullcontext():
                validate_project_manifest(config.get('project'), config['manifest'], training_root)
                return self._start_locked(config)
    def _start_locked(self,config):
        if self.thread and self.thread.is_alive(): raise RuntimeError('Training is already running.')
        self.stop_event.clear(); self._update(running=True,step=0,total=0,message='Starting experimental LoRA training…',output='')
        def worker():
            try:
                # Keep the GUI and dataset-preparation tab usable when the
                # optional CUDA/PyTorch runtime is unavailable or broken.
                from .trainer import train_lora
                result=train_lora(config,self._update,self.stop_event)
                self._update(running=False,output=result.get('output',''),message='Training stopped.' if result.get('stopped') else 'Training complete.'); _play_done_sound()
            except Exception as exc:
                import traceback; traceback.print_exc(); self._update(running=False,message=f'Training failed: {type(exc).__name__}: {exc}')
                # A failed CUDA step can otherwise leave the large GPT/Whisper
                # teacher cached in the GUI process, making the next retry
                # fail with an unrelated out-of-memory error.
                try:
                    import gc; gc.collect()
                    import torch
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                except Exception:
                    pass
        self.thread=threading.Thread(target=worker,daemon=True); self.thread.start(); return 'Training started.'
    def stop(self): self.stop_event.set(); self._update(message='Stop requested…'); return 'Stop requested.'
    def _stop_tensorboard_server(self):
        process=self.tensorboard
        self.tensorboard=None
        self.tensorboard_logdir=None
        if process is None or process.poll() is not None: return
        try:
            process.terminate()
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        except Exception:
            pass

    def start_tensorboard(self,logdir,port=6006,force_restart=False):
        logdir=str(Path(logdir).resolve())
        url=f'http://127.0.0.1:{port}'
        with self.lock:
            running=self.tensorboard is not None and self.tensorboard.poll() is None
            same_logdir=running and self.tensorboard_logdir == logdir
            if running and (force_restart or not same_logdir):
                # Only the TensorBoard viewer process is replaced here. The
                # event files are never touched, so a training worker keeps
                # writing to its existing log directory uninterrupted.
                self._stop_tensorboard_server()
                same_logdir=False
            if not same_logdir:
                self.tensorboard=subprocess.Popen([
                    sys.executable,'-m','tensorboard.main',
                    '--logdir',logdir,'--port',str(port),'--host','127.0.0.1',
                    '--reload_interval','1','--reload_multifile','True',
                ])
                self.tensorboard_logdir=logdir
                message=f'TensorBoard started on {url}; live log reload enabled.'
            else:
                message=f'TensorBoard is live on {url}; monitoring updated event files.'
        # Re-open the same stable URL when the user resumes or returns to the
        # training tab. This does not restart or rewrite any training log.
        webbrowser.open(url)
        return message

    def reload_tensorboard(self,logdir,port=6006):
        """Force-refresh the selected training workspace viewer only."""
        return self.start_tensorboard(logdir,port,force_restart=True)
TRAINING_MANAGER=TrainingManager()

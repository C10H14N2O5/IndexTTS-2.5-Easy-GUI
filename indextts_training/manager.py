from __future__ import annotations
import json, threading, time, subprocess, sys, webbrowser
from pathlib import Path


class TrainingManager:
    def __init__(self):
        self.lock=threading.RLock(); self.thread=None; self.stop_event=threading.Event(); self.tensorboard=None; self.tensorboard_logdir=None
        # The GUI installs a fake trainer for development; None = indextts_training.trainer.train_lora.
        self.train_fn=None
        # Optional completion hook (the web GUI notifies browsers instead of playing a sound on the server).
        self.on_finished=None
        # Called only after acquiring the shared activity lease (e.g. evict inference weights).
        self.before_start=None
        self.state={'running':False,'failed':False,'step':0,'total':0,'schedule_total':0,'epoch':0,'loss':None,'elapsed':0.0,'eta':0.0,'message':'Idle.','output':'',
                    'started_at':None,'finished_at':None,'output_dir':'','project':'','adapter':''}
    def _update(self,**kw):
        with self.lock: self.state.update({k:v for k,v in kw.items() if v is not None})
    def snapshot(self):
        with self.lock: return dict(self.state)
    def start(self,config):
        from .dataset import DATASET_LOCK, load_project, validate_project_manifest, validate_validation_manifest
        from .proofread import directory_lock
        with DATASET_LOCK:
            training_root = config.get('training_root', 'training')
            project = load_project(config.get('project'), training_root)
            with directory_lock(project.get('source_folder')):
                validate_project_manifest(config.get('project'), config['manifest'], training_root)
                run_config = dict(config)
                overrides = {}
                if 'validation_enabled' in run_config:
                    overrides['enabled'] = run_config.pop('validation_enabled')
                if 'validation_interval_steps' in run_config:
                    overrides['interval_steps'] = run_config.pop('validation_interval_steps')
                run_config['validation'] = validate_validation_manifest(
                    config.get('project'), config['manifest'], training_root, **overrides)
                return self._start_locked(run_config)
    def _start_locked(self,config):
        if self.thread and self.thread.is_alive(): raise RuntimeError('Training is already running.')
        from .activity import acquire, release
        activity_token = acquire('training')
        self.stop_event.clear()
        with self.lock:
            # Assign directly: _update() ignores None, which used to leave the previous
            # run's loss/epoch/ETA visible until the first step of the new run.
            self.state.update(running=True,failed=False,step=0,total=0,schedule_total=0,epoch=0,loss=None,elapsed=0.0,eta=0.0,
                              message='Starting experimental LoRA training…',output='',started_at=time.time(),finished_at=None,
                              output_dir=str(config.get('output_dir') or ''),project=str(config.get('project') or ''),
                              adapter=Path(str(config.get('output_dir') or '')).name)
        def worker():
            outcome='failed'
            try:
                # Keep the GUI and dataset-preparation tab usable when the
                # optional CUDA/PyTorch runtime is unavailable or broken.
                train_lora=self.train_fn
                if train_lora is None:
                    from .trainer import train_lora
                result=train_lora(config,self._update,self.stop_event)
                outcome='stopped' if result.get('stopped') else 'succeeded'
                if result.get('stopped'):
                    message='Training stopped.'
                elif result.get('early_stopped'):
                    message=f"Training finished early at step {result.get('step')} (LR schedule {result.get('schedule_total') or result.get('total')})."
                else:
                    message='Training complete.'
                self._update(running=False,output=result.get('output',''),finished_at=time.time(),message=message)
            except Exception as exc:
                import traceback; traceback.print_exc(); self._update(running=False,failed=True,finished_at=time.time(),message=f'Training failed: {type(exc).__name__}: {exc}')
                # A failed CUDA step can otherwise leave the large GPT/Whisper
                # teacher cached in the GUI process, making the next retry
                # fail with an unrelated out-of-memory error.
                try:
                    import gc; gc.collect()
                    import torch
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                except Exception:
                    pass
            finally:
                release(activity_token)
                hook=self.on_finished
                if hook is not None:
                    try: hook(outcome,self.snapshot())
                    except Exception: pass
        try:
            if self.before_start is not None:
                self.before_start()
            if config.get('output_dir'):
                from .dataset import _atomic_write
                saved = dict(config)
                if config.get('validation'):
                    saved['validation'] = {key: value for key, value in config['validation'].items()
                                           if key != 'rows'}
                    saved['validation']['samples'] = len(config['validation']['rows'])
                _atomic_write(Path(config['output_dir']) / 'training_config.json',
                              json.dumps(saved, indent=2).encode('utf-8'))
            self.thread=threading.Thread(target=worker,daemon=True)
            self.thread.start()
        except Exception:
            self._update(running=False)
            release(activity_token)
            raise
        return 'Training started.'
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

    def start_tensorboard(self,logdir,port=6006,force_restart=False,open_browser=True):
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
        # The web GUI opens the URL in the user's browser tab instead (open_browser=False).
        if open_browser:
            webbrowser.open(url)
        return message

    def reload_tensorboard(self,logdir,port=6006,open_browser=True):
        """Force-refresh the selected training workspace viewer only."""
        return self.start_tensorboard(logdir,port,force_restart=True,open_browser=open_browser)
TRAINING_MANAGER=TrainingManager()

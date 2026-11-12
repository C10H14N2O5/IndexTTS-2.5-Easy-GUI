import type { Job } from '@/api/live';
import type { AdapterFolder, Checkpoint, PosthocItem, PosthocResults, Project, TrainingConfig, TrainingStatus } from './api';
export const trainingDefaults: TrainingConfig = {
  adapter_name: '', vram_preset: '24 GB', mode: 'Steps', training_steps: 1000, training_epochs: 20, save_steps: 100, save_epochs: 2, early_stop_enabled: false, early_stop_step: 500,
  auto_stop_enabled: false, auto_stop_patience_steps: 200, auto_stop_min_delta: .005, auto_stop_watch_interval_steps: 25,
  lora_rank: 16, lora_alpha: 32, lora_dropout: .05, learning_rate: .00004, micro_batch: 1, grad_accum: 8, grad_clip: .5, seed: 1234,
  resume_checkpoint: '', eval_enabled: true, eval_text: 'This is an IndexTTS 2.5 training preview.', eval_reference: '', eval_reference_transcript: '', eval_language: 'English', eval_whisper_model: 'sensevoice-small-q8', eval_whisper_language: 'Auto-detect',
  semantic_loss_weight: 0, reference_text_conditioning: false, phonetic_ctc_enabled: false, phonetic_ctc_weight: 0, legacy_phonetic_ctc_enabled: false,
  differentiable_asr_enabled: false, differentiable_asr_weight: 0, differentiable_asr_model: 'large-v3', asr_cfm_steps: 4, asr_cfm_temperature: 1,
};
export function projectFixture(): Project { return {
  project_name: 'voice', source_folder: 'D:/corpus', manifest: 'D:/prepared/manifest.jsonl', language: 'ZH', modified: 1234567890,
  training: { ...trainingDefaults, adapter_name: 'voice' }, stats: { valid: 12, total_seconds: 60 },
  validation: { source_folder: '', manifest: '', enabled: false, interval_steps: null, dense_enabled: false, dense_start_step: null, dense_end_step: null, dense_interval_steps: null, stats: {} }, validation_suggested_folder: 'D:/validation/voice',
  dataset_status: { stale: false, managed: true, manifest: 'D:/prepared/manifest.jsonl', status: 'ready', message: 'Training manifest is up to date.', stats: {} },
  validation_status: { stale: true, managed: false, manifest: '', status: 'missing', message: 'Validation manifest is missing.', stats: {} },
}; }
export const adapterFolders: AdapterFolder[] = [
  { name: 'old_voice', path: 'D:/out/old_voice', checkpoints: 1, has_final: false, has_best: false, modified: 100, is_project_adapter: false },
  { name: 'voice', path: 'D:/out/voice', checkpoints: 2, has_final: true, has_best: true, modified: 200, is_project_adapter: true },
];
export function checkpointsFixture(adapter: string): Checkpoint[] {
  if (adapter === 'old_voice') return [{ name: 'checkpoint-step-000050', path: 'D:/out/old_voice/checkpoint-step-000050', adapter, kind: 'checkpoint', step: 50, modified: 100, resumable: true, early_stopped: false, auto_stopped: false }];
  return [
    { name: 'checkpoint-step-000100', path: 'D:/out/voice/checkpoint-step-000100', adapter, kind: 'checkpoint', step: 100, modified: 150, resumable: true, early_stopped: false, auto_stopped: false },
    { name: 'checkpoint-step-000200', path: 'D:/out/voice/checkpoint-step-000200', adapter, kind: 'checkpoint', step: 200, modified: 200, resumable: true, early_stopped: true, auto_stopped: false },
    { name: 'final_adapter', path: 'D:/out/voice/final_adapter', adapter, kind: 'final', step: 200, modified: 200, resumable: false, early_stopped: true, auto_stopped: false },
  ];
}
/** Present only when a test opts in, so other output tests keep their exact checkpoint lists. */
export const bestCheckpoint: Checkpoint = { name: 'best_adapter', path: 'D:/out/voice/best_adapter', adapter: 'voice', kind: 'best', step: 150, modified: 210, resumable: true, early_stopped: false, auto_stopped: false, validation_loss: 1.04321 };
export function posthocFixture(adapter: string): PosthocResults {
  const item = (name: string, kind: PosthocItem['kind'], step: number, loss: number, status: PosthocItem['status'] = 'current'): PosthocItem => ({ name, kind, step, loss, validation_fingerprint: 'abcdef1234567890', samples: 6, computed_at: 100, status, same_as: null });
  return { project: 'voice', adapter, training_fingerprint: '0123456789abcdef', validation: { ready: true, message: 'Validation manifest is ready.', fingerprint: 'abcdef1234567890', samples: 6, manifest: 'D:/prepared/validation.jsonl' },
    items: adapter === 'voice' ? [item('base_model', 'baseline', 0, 2.5), item('checkpoint-step-000100', 'checkpoint', 100, 1.23456), item('checkpoint-step-000200', 'checkpoint', 200, 1.1, 'outdated')] : [] };
}
export const posthocJob: Job = { id: 'posthoc1', kind: 'validation_posthoc', title: 'Computing validation loss.', state: 'running', progress: .25, stage: 'Validation checkpoint-step-000200: sample 1/6', started_at: Date.now() / 1000, finished_at: null, cancellable: true, cancel_requested: false, gpu: true, meta: { project: 'voice', adapter: 'voice', targets: ['checkpoint-step-000200'] } };
export const idleTraining: TrainingStatus = { running: false, failed: false, stopped: false, status: 'idle', step: 0, total: 0, schedule_total: 0, auto_stop: null, auto_stopped: false, epoch: 0, loss: null, elapsed: 0, eta: 0, message: '', output: '', output_dir: '', project: '', adapter: '', started_at: null, finished_at: null };

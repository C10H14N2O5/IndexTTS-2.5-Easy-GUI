import type { Project, TrainingConfig, TrainingStatus } from './api';
export const trainingDefaults: TrainingConfig = {
  adapter_name: '', vram_preset: '24 GB', mode: 'Steps', training_steps: 1000, training_epochs: 20, save_steps: 100, save_epochs: 2,
  lora_rank: 16, lora_alpha: 32, lora_dropout: .05, learning_rate: .00004, micro_batch: 1, grad_accum: 8, grad_clip: .5, seed: 1234,
  resume_checkpoint: '', eval_enabled: true, eval_text: 'This is an IndexTTS 2.5 training preview.', eval_reference: '', eval_reference_transcript: '', eval_language: 'English', eval_whisper_model: 'sensevoice-small-q8', eval_whisper_language: 'Auto-detect',
  semantic_loss_weight: 0, reference_text_conditioning: false, phonetic_ctc_enabled: false, phonetic_ctc_weight: 0, legacy_phonetic_ctc_enabled: false,
  differentiable_asr_enabled: false, differentiable_asr_weight: 0, differentiable_asr_model: 'large-v3', asr_cfm_steps: 4, asr_cfm_temperature: 1,
};
export function projectFixture(): Project { return {
  project_name: 'voice', source_folder: 'D:/corpus', manifest: 'D:/prepared/manifest.jsonl', language: 'ZH', modified: 1234567890,
  training: { ...trainingDefaults, adapter_name: 'voice' }, stats: { valid: 12, total_seconds: 60 },
  validation: { source_folder: '', manifest: '', enabled: false, interval_steps: null, stats: {} }, validation_suggested_folder: 'D:/validation/voice',
  dataset_status: { stale: false, managed: true, manifest: 'D:/prepared/manifest.jsonl', status: 'ready', message: 'Training manifest is up to date.', stats: {} },
  validation_status: { stale: true, managed: false, manifest: '', status: 'missing', message: 'Validation manifest is missing.', stats: {} },
}; }
export const idleTraining: TrainingStatus = { running: false, failed: false, stopped: false, status: 'idle', step: 0, total: 0, epoch: 0, loss: null, elapsed: 0, eta: 0, message: '', output: '', output_dir: '', project: '', adapter: '', started_at: null, finished_at: null };

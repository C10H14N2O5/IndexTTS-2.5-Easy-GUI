import type { Bootstrap } from '@/app/bootstrap';
import { GENERATION_DEFAULTS } from './model';
export function testBootstrap(): Bootstrap {
  return {
    version: '2.0.0', fake_engine: true,
    settings: { ui_language: 'en_US', theme: 'system', completion_sound: false, notifications: false, use_bf16: true, use_gpt_latent: false, use_torch_compile: false, use_flash_accel: false },
    capabilities: { triton: { ok: false, detail: 'No Triton' }, flash_attn: { ok: false, detail: 'No FlashAttention' } },
    system: { model_present: false, model_dir: 'models', model: { loaded: false }, vram: { allocated_gb: 0, reserved_gb: 0, total_gb: 0 }, cuda: { available: false }, fake_engine: true },
    catalog: { languages: [{ code: 'ZH', label: 'Chinese' }, { code: 'EN', label: 'English' }], default_language: 'ZH', emotions: ['Happy', 'Angry', 'Sad', 'Afraid', 'Disgusted', 'Melancholic', 'Surprised', 'Calm'], emotion_modes: ['Same as voice reference', 'Emotion reference audio', 'Emotion vector', 'Emotion description'], emotion_presets: { 'Neutral / balanced': [0, 0, 0, 0, 0, 0, 0, 0.55] }, default_preset: 'Neutral / balanced', dialogue_emotion_sources: ['Emotion Vector Preset', 'Emotion Description'], dialogue_max_turns: 12, chunk_modes: ['None', 'Lines'], vram_presets: ['24 GB'], lora_ranks: [16], synthesis_params: Object.fromEntries(Object.entries(GENERATION_DEFAULTS).filter(([, value]) => typeof value === 'number').map(([key, value]) => [key, { min: 0, max: 3000, step: 0.1, default: Number(value) }])), synthesis_flags: { do_sample: true } },
    training_defaults: {},
  };
}

from __future__ import annotations
import functools
import ctypes
import hashlib
import json, math, os, re, time, shutil, subprocess, unicodedata
import threading
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio


PHONETIC_BLANK = 0
PHONETIC_SYMBOLS = list('abcdefghijklmnopqrstuvwxyz') + [
    'ɾ', 'θ', 'ɲ', 'ʝ', 'β', 'ð', 'ɣ', 'ʎ', 'ʃ', 'ʒ', 'ŋ', 'ʔ',
    'ɐ', 'ə', 'ɤ', 'ø', 'y', 'ç', 'χ', 'ɸ', 'ʂ', 'ʐ', 'ɬ', 'ɹ',
    'ɻ', 'ɪ', 'ʊ', 'ɛ', 'ɔ', 'sil',
]
PHONETIC_TO_ID = {symbol: index + 1 for index, symbol in enumerate(PHONETIC_SYMBOLS)}
_ESPEAK_NATIVE = None
_ESPEAK_NATIVE_LOCK = threading.RLock()


def _load_espeak_native():
    """Load the bundled eSpeak NG library shipped by espeakng-loader."""
    global _ESPEAK_NATIVE
    with _ESPEAK_NATIVE_LOCK:
        if _ESPEAK_NATIVE is not None:
            return _ESPEAK_NATIVE
        try:
            from espeakng_loader import get_data_path, get_library_path, make_library_available

            make_library_available()
            library = ctypes.CDLL(get_library_path())
            library.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
            library.espeak_Initialize.restype = ctypes.c_int
            sample_rate = library.espeak_Initialize(0, 0, get_data_path().encode("utf-8"), 0)
            if sample_rate <= 0:
                return None
            library.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
            library.espeak_SetVoiceByName.restype = ctypes.c_int
            library.espeak_TextToPhonemes.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int,
            ]
            library.espeak_TextToPhonemes.restype = ctypes.c_char_p
            _ESPEAK_NATIVE = library
            return library
        except Exception:
            return None


def _native_espeak_ipa(text, language):
    library = _load_espeak_native()
    if library is None:
        return ""
    try:
        with _ESPEAK_NATIVE_LOCK:
            if library.espeak_SetVoiceByName(_espeak_voice(language).encode("utf-8")) != 0:
                return ""
            buffer = ctypes.create_string_buffer(str(text).encode("utf-8"))
            text_ptr = ctypes.c_void_p(ctypes.addressof(buffer))
            result = library.espeak_TextToPhonemes(ctypes.byref(text_ptr), 0, 2)
            return result.decode("utf-8", errors="ignore") if result else ""
    except (OSError, UnicodeError, ctypes.ArgumentError):
        return ""


def _espeak_executable():
    configured = os.environ.get('ESPEAK_NG_PATH', '').strip()
    candidates = [
        configured,
        shutil.which('espeak-ng'),
        r'C:\Program Files\eSpeak NG\espeak-ng.exe',
        r'C:\Program Files (x86)\eSpeak NG\espeak-ng.exe',
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def _espeak_voice(language):
    value = str(language or 'es').lower().replace('_', '-')
    return {
        'es': 'es', 'spa': 'es', 'spanish': 'es',
        'en': 'en-us', 'eng': 'en-us', 'english': 'en-us',
        'ja': 'ja', 'jpn': 'ja', 'zh': 'cmn', 'zho': 'cmn',
        'ar': 'ar', 'ara': 'ar',
    }.get(value, value or 'es')


def _parse_espeak_ipa(output):
    # eSpeak returns stress/length/tie marks in addition to IPA phones. The
    # stress marks are intentionally dropped so CTC learns the phoneme
    # sequence first, without forcing a separate prosody alignment objective.
    ignored = set('ˈˌːˑ͜͡')
    tokens = []
    for char in str(output or ''):
        if char in ignored or char in '\r\n':
            continue
        if char.isspace():
            if tokens and tokens[-1] != 'sil':
                tokens.append('sil')
            continue
        if char in PHONETIC_TO_ID:
            tokens.append(char)
    while tokens and tokens[-1] == 'sil':
        tokens.pop()
    return tuple(tokens)


@functools.lru_cache(maxsize=8192)
def _espeak_phonemes(text, language):
    text = str(text or '').strip()
    if not text:
        return ()
    native_ipa = _native_espeak_ipa(text, language)
    if native_ipa:
        return _parse_espeak_ipa(native_ipa)
    executable = _espeak_executable()
    if not executable:
        return ()
    try:
        result = subprocess.run(
            [executable, '-q', '--ipa', '-v', _espeak_voice(language), text],
            capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if result.returncode != 0 or not result.stdout.strip():
        return ()
    return _parse_espeak_ipa(result.stdout)


@functools.lru_cache(maxsize=8192)
def _phonetic_tokens(text, language='es'):
    """Create deterministic phonetic-like targets without an external G2P binary."""
    ipa_tokens = _espeak_phonemes(str(text or ''), str(language or 'es'))
    if ipa_tokens:
        return ipa_tokens

    raw = unicodedata.normalize('NFKD', str(text or '')).lower()
    raw = ''.join(ch for ch in raw if not unicodedata.combining(ch))
    raw = re.sub(r'[^a-z\s]', ' ', raw)
    raw = re.sub(r'\s+', ' ', raw).strip()
    if not raw:
        return ()
    if str(language).lower() not in ('es', 'spa', 'spanish'):
        return tuple(ch for ch in raw if ch in PHONETIC_TO_ID)

    out = []
    i = 0
    while i < len(raw):
        if raw[i].isspace():
            i += 1
            continue
        pair = raw[i:i + 2]
        if pair in ('ch', 'll', 'rr'):
            out.append(pair); i += 2; continue
        if pair == 'qu':
            out.append('k'); i += 2; continue
        if pair == 'gu' and i + 2 < len(raw) and raw[i + 2] in 'ei':
            out.append('g'); i += 2; continue
        ch = raw[i]
        nxt = raw[i + 1] if i + 1 < len(raw) else ''
        if ch == 'c':
            out.append('k' if nxt in 'aou' else 's')
        elif ch == 'z':
            out.append('s')
        elif ch == 'j':
            out.append('x')
        elif ch == 'g' and nxt in 'ei':
            out.append('x')
        elif ch == 'h':
            pass
        elif ch == 'v':
            out.append('b')
        elif ch in PHONETIC_TO_ID:
            out.append(ch)
        i += 1
    return tuple(out)


def _phonetic_target(text, language, device):
    ids = [PHONETIC_TO_ID[token] for token in _phonetic_tokens(text, language) if token in PHONETIC_TO_ID]
    return torch.tensor(ids, dtype=torch.long, device=device)


def _read_manifest(path, fingerprint=None):
    data = Path(path).read_bytes()
    if fingerprint and hashlib.sha256(data).hexdigest() != fingerprint:
        raise RuntimeError('Training manifest changed after validation preflight. Start training again.')
    return [json.loads(line) for line in data.decode('utf-8-sig').splitlines() if line.strip()]

def _safe_audio(path, sr=16000, max_seconds=15.0):
    try:
        import soundfile as sf
        import numpy as np
        data, source_sr = sf.read(path, dtype='float32', always_2d=True)
        wav = torch.from_numpy(np.asarray(data).T.copy())
    except Exception:
        # Keep a compatibility fallback for formats unavailable to libsndfile.
        wav, source_sr=torchaudio.load(path)
    wav=wav.mean(0,keepdim=True)
    if source_sr!=sr: wav=torchaudio.functional.resample(wav,source_sr,sr)
    return wav[:,:int(sr*max_seconds)]

def _prepare_sample(tts,item,device,include_auxiliary_conditioning=True):
    wav16=_safe_audio(item['audio'],16000)
    inputs=tts.extract_features(wav16, sampling_rate=16000, return_tensors='pt')
    feats=inputs['input_features'].to(device); mask=inputs['attention_mask'].to(device)
    with torch.no_grad():
        semantic=tts.get_emb(feats,mask)
        codes=tts.semantic_codec.quantize(semantic)[0].long()
        if codes.ndim==3: codes=codes.squeeze(1)
        if codes.ndim==1: codes=codes.unsqueeze(0)
        fb=torchaudio.compliance.kaldi.fbank(
            wav16.to(device), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        fb=fb-fb.mean(dim=0,keepdim=True)

        # CAMPPlus remains in its native dtype (normally FP32). IndexTTS 2.5 loads
        # the GPT conditioning layers in BF16 when use_bf16=True, so condition
        # tensors must be converted only at the GPT boundary.
        camp_style=tts.campplus_model(fb.unsqueeze(0))

        spk_proj_dtype=tts.gpt.spk_emb_proj.weight.dtype
        spk_proj_device=tts.gpt.spk_emb_proj.weight.device
        style_for_gpt=camp_style.to(device=spk_proj_device,dtype=spk_proj_dtype)
        spk=tts.gpt.spk_emb_proj(style_for_gpt).unsqueeze(1)

        lens=torch.tensor([semantic.shape[1]],device=device)

        # The emotion-conditioning stack is part of the BF16 GPT runtime as well.
        # Keep the semantic representation untouched for codec quantization and
        # cast only the copy consumed by the GPT emotion encoder.
        try:
            emo_param=next(tts.gpt.emo_conditioning_encoder.parameters())
            emo_input=semantic.to(device=emo_param.device,dtype=emo_param.dtype)
        except (AttributeError, StopIteration):
            emo_input=semantic.to(device=spk.device,dtype=spk.dtype)

        emo=tts.gpt.get_emovec(emo_input,lens.to(emo_input.device))
        emo=emo.to(device=spk.device,dtype=spk.dtype)

        if include_auxiliary_conditioning:
            # Keep the complete reference conditioning required by the
            # optional differentiable CFM/S2Mel path. These tensors are frozen
            # conditioning inputs; gradients flow only through the generated
            # target path.
            wav22=torchaudio.functional.resample(wav16,16000,22050)
            ref_mel=tts.mel_fn(wav22.to(device).float()).detach()
            ref_target_lengths=torch.tensor([ref_mel.shape[-1]],dtype=torch.long,device=device)
            s2mel_prompt=tts.s2mel.models['length_regulator'](
                semantic.detach().float(), ylens=ref_target_lengths, n_quantizers=3, f0=None
            )[0].detach()
            auxiliary_style=camp_style.detach().float()
        else:
            s2mel_prompt=None
            ref_mel=None
            auxiliary_style=None
    lang=str(item.get('language','EN')).lower(); prefix=f'<|{lang}|> '
    # Keep both target and reference transcripts inside the positional window
    # used by the IndexTTS GPT.  Long sidecars should be truncated safely at
    # the token boundary instead of failing later in an embedding lookup.
    max_text_tokens=max(1,int(getattr(tts.gpt,'max_text_tokens',120)))
    toks=tts.tokenizer.encode(prefix+item['text'],allowed_special='all')[:max_text_tokens]
    text=torch.tensor(toks,dtype=torch.long,device=device).unsqueeze(0)
    reference_source=str(item.get('reference_text') or item.get('text') or '').strip()
    reference_toks=tts.tokenizer.encode(prefix+reference_source,allowed_special='all')[:max_text_tokens] if reference_source else []
    reference_text=torch.tensor(reference_toks,dtype=torch.long,device=device).unsqueeze(0)
    from indextts.utils.tokenizer import lang_to_token
    lang_id=torch.tensor([lang_to_token(lang)],dtype=torch.long,device=device)
    phonetic_target=_phonetic_target(item.get('text',''),lang,device) if include_auxiliary_conditioning else torch.empty(0,dtype=torch.long,device=device)
    return (
        text, reference_text, spk, emo, codes, lang_id, phonetic_target,
        s2mel_prompt, ref_mel, auxiliary_style, str(item.get('text','')),
    )

def _soft_semantic_decode(tts, acoustic_logits, temperature=1.0):
    """Decode expected codec embeddings while preserving gradients to GPT logits."""
    codec=tts.semantic_codec
    quantizer=codec.quantizer.quantizers[0]
    steps=acoustic_logits.shape[-1]
    logits=acoustic_logits[:, :int(codec.codebook_size), :steps]
    probs=F.softmax(logits.float()/max(0.05,float(temperature)),dim=1)
    codebook=quantizer.codebook.weight.to(device=probs.device,dtype=probs.dtype)
    expected=torch.einsum('bkt,kd->btd',probs,codebook)
    quantized=quantizer.out_project(expected.transpose(1,2))
    decoded=codec.decoder(quantized)
    if codec.downsample_scale is not None and codec.downsample_scale > 1:
        decoded=codec.up(F.interpolate(decoded.transpose(1,2),scale_factor=2,mode='nearest')).transpose(1,2)
    return decoded


def _load_differentiable_asr(config, device):
    """Load a frozen, gradient-through-input Whisper teacher for training."""
    if not bool(config.get('differentiable_asr_enabled', False)):
        return None
    import whisper

    model_name=str(config.get('differentiable_asr_model','large-v3') or 'large-v3')
    cache_dir=str(config.get('asr_cache_dir') or (Path(config['model_dir'])/'hf_cache'/'whisper'))
    Path(cache_dir).mkdir(parents=True,exist_ok=True)
    print(f'[training] Loading differentiable Whisper ASR teacher: {model_name}',flush=True)
    try:
        asr=whisper.load_model(model_name,device=device,download_root=cache_dir)
    except Exception as exc:
        raise RuntimeError(
            f'Could not load differentiable Whisper teacher {model_name!r}. '
            f'It is downloaded to {cache_dir} on first use; verify network/cache access. '
            f'{type(exc).__name__}: {exc}'
        ) from exc
    asr.eval()
    for parameter in asr.parameters():
        parameter.requires_grad=False
    return asr


def _whisper_teacher_targets(asr, text, language, device):
    from whisper.tokenizer import get_tokenizer

    language=str(language or 'es').lower()
    if not asr.is_multilingual:
        language=None
    tokenizer=get_tokenizer(
        asr.is_multilingual,
        num_languages=getattr(asr,'num_languages',99),
        language=language,
        task='transcribe',
    )
    prefix=list(tokenizer.sot_sequence_including_notimestamps)
    body=tokenizer.encode(str(text or '').strip())
    max_body=max(1,int(asr.dims.n_text_ctx)-len(prefix)-1)
    sequence=prefix+body[:max_body]+[tokenizer.eot]
    return (
        torch.tensor(sequence[:-1],dtype=torch.long,device=device).unsqueeze(0),
        torch.tensor(sequence[1:],dtype=torch.long,device=device).unsqueeze(0),
    )


def _whisper_mel_from_waveform(waveform, asr, device):
    import whisper

    if waveform.ndim==1:
        waveform=waveform.unsqueeze(0)
    waveform16=torchaudio.functional.resample(waveform.float(),22050,16000)
    mel=whisper.log_mel_spectrogram(
        waveform16,
        n_mels=int(asr.dims.n_mels),
        device=device,
    )
    # Whisper's encoder downsamples the 80-bin input by 2; ``n_audio_ctx``
    # describes the post-convolution sequence, so the input mel has twice
    # that many frames (3000 for the standard models).
    target_frames=int(asr.dims.n_audio_ctx)*2
    if mel.shape[-1]<target_frames:
        mel=F.pad(mel,(0,target_frames-mel.shape[-1]))
    else:
        mel=mel[...,:target_frames]
    return mel


def _differentiable_asr_loss(
    tts, asr, acoustic_logits, s2mel_prompt, ref_mel, style, target_text,
    language='es', cfm_steps=4, temperature=1.0,
):
    """Run GPT -> soft codec -> length regulator -> CFM -> vocoder -> Whisper.

    The Whisper weights are frozen, but its input remains attached to the
    graph. Consequently the ASR cross-entropy gradient travels through the
    differentiable log-mel transform, BigVGAN, CFM Euler steps, S2Mel length
    regulator and finally into the GPT acoustic logits.
    """
    if not str(target_text or '').strip():
        return torch.zeros((),device=acoustic_logits.device,dtype=torch.float32)

    soft_features=_soft_semantic_decode(tts,acoustic_logits,temperature)
    mel_ratio=float(1.72)
    target_frames=max(1,int(round(soft_features.shape[1]*mel_ratio)))
    target_lengths=torch.tensor([target_frames],dtype=torch.long,device=soft_features.device)
    length_regulator=tts.s2mel.models['length_regulator']
    cond=length_regulator(
        soft_features,
        ylens=target_lengths,
        n_quantizers=3,
        f0=None,
    )[0]
    prompt=s2mel_prompt.to(device=cond.device,dtype=cond.dtype)
    ref_mel=ref_mel.to(device=cond.device,dtype=cond.dtype)
    style=style.to(device=cond.device,dtype=cond.dtype)
    cat_condition=torch.cat([prompt,cond],dim=1)
    full_lengths=torch.tensor([cat_condition.shape[1]],dtype=torch.long,device=cond.device)
    cfm=tts.s2mel.models['cfm']
    generated_mel=cfm.differentiable_inference(
        cat_condition,
        full_lengths,
        ref_mel,
        style,
        None,
        n_timesteps=max(1,int(cfm_steps)),
        temperature=float(temperature),
        inference_cfg_rate=0.0,
    )
    generated_mel=generated_mel[:,:,prompt.shape[1]:]
    if generated_mel.requires_grad:
        from torch.utils.checkpoint import checkpoint
        waveform=checkpoint(tts.bigvgan,generated_mel.float(),use_reentrant=False).squeeze(1)
    else:
        waveform=tts.bigvgan(generated_mel.float()).squeeze(1)
    asr_mel=_whisper_mel_from_waveform(waveform,asr,acoustic_logits.device)
    decoder_inputs,targets=_whisper_teacher_targets(
        asr,target_text,language,acoustic_logits.device
    )
    decoder_inputs=decoder_inputs.expand(asr_mel.shape[0],-1)
    targets=targets.expand(asr_mel.shape[0],-1)
    asr_dtype=next(asr.parameters()).dtype
    # Keep the frozen teacher numerically stable even though the surrounding
    # GPT/CFM step uses BF16 autocast.
    with torch.autocast(device_type=asr_mel.device.type,enabled=False):
        teacher_mel=asr_mel.to(dtype=asr_dtype)
        # The teacher is frozen, but its input still requires gradients.  A
        # normal Whisper forward retains a very large encoder/decoder graph
        # for those input gradients; checkpointing the two blocks keeps the
        # full differentiable ASR objective usable on 24 GB GPUs.
        if teacher_mel.requires_grad and hasattr(asr, 'encoder') and hasattr(asr, 'decoder'):
            from torch.utils.checkpoint import checkpoint
            audio_features=checkpoint(asr.encoder, teacher_mel, use_reentrant=False)
            logits=checkpoint(asr.decoder, decoder_inputs, audio_features, use_reentrant=False)
        else:
            logits=asr(teacher_mel,decoder_inputs)
    return F.cross_entropy(
        logits.float().reshape(-1,logits.shape[-1]),
        targets.reshape(-1),
        reduction='mean',
    )


def _loss(tts,text,reference_text,spk,emo,codes,lang_id,phonetic_target,semantic_weight=0.0,
          reference_text_conditioning=False,phonetic_weight=0.0,phonetic_temperature=1.0,
          asr_model=None,asr_weight=0.0,asr_language='es',asr_cfm_steps=4,
          asr_temperature=1.0,s2mel_prompt=None,ref_mel=None,style=None,target_text=''):
    g=tts.gpt
    text_l=torch.tensor([text.shape[1]],device=text.device); code_l=torch.tensor([codes.shape[1]],device=text.device)
    text_i=F.pad(text,(0,1),value=g.stop_text_token); code_i=F.pad(codes,(0,1),value=g.stop_mel_token)
    text_i,text_t=g.build_aligned_inputs_and_targets(text_i,g.start_text_token,g.stop_text_token)
    code_i,code_t=g.build_aligned_inputs_and_targets(code_i,g.start_mel_token,g.stop_mel_token)
    # Conditioning tensors must match the GPT embedding dtype. This is
    # especially important when the fixed runtime loads IndexTTS 2.5 in BF16.
    gpt_dtype=g.text_embedding.weight.dtype
    spk=spk.to(device=text.device,dtype=gpt_dtype)
    emo=emo.to(device=text.device,dtype=gpt_dtype)
    if reference_text_conditioning:
        reference_condition = g.get_reference_text_conditioning(reference_text, lang_id)
        if reference_condition is not None:
            reference_condition = reference_condition.to(device=spk.device,dtype=gpt_dtype)
            spk = spk + reference_condition.unsqueeze(1)
    cond=torch.cat(
        (
            spk+emo.unsqueeze(1),
            torch.zeros(spk.size(0),2,spk.size(2),device=spk.device,dtype=gpt_dtype),
        ),
        1,
    )
    te=g.text_embedding(text_i)+g.text_pos_embedding(text_i)
    te=te+g.lang_embedding(lang_id).unsqueeze(1)
    me=g.mel_embedding(code_i)+g.mel_pos_embedding(code_i)
    text_logits,logits=g.get_logits(cond,te,g.text_head,me,g.mel_head,get_attns=False,return_latent=False)
    target=code_t
    n=min(logits.shape[-1],target.shape[-1]); logits=logits[:,:,:n]; target=target[:,:n]
    mel_loss=F.cross_entropy(logits.float(),target.long(),label_smoothing=0.0)
    if float(semantic_weight) > 0.0:
        text_target=text_t
        text_n=min(text_logits.shape[-1],text_target.shape[-1])
        text_loss=F.cross_entropy(
            text_logits[:,:,:text_n].float(), text_target[:,:text_n].long(), label_smoothing=0.0
        )
    else:
        # Do not create or report an auxiliary text-prediction objective in
        # the acoustic-only profile. The target text remains present only as
        # the mandatory conditioning input for IndexTTS's native mel loss.
        text_loss=torch.zeros((),device=text.device,dtype=mel_loss.dtype)
    phonetic_loss=torch.zeros((),device=text.device,dtype=mel_loss.dtype)
    asr_loss=torch.zeros((),device=text.device,dtype=mel_loss.dtype)
    if float(phonetic_weight) > 0.0 and phonetic_target.numel() > 0:
        phonetic_head=getattr(g,'phonetic_head',None)
        if phonetic_head is None:
            phonetic_head=getattr(g.gpt,'phonetic_head',None)
        if phonetic_head is None:
            raise RuntimeError('Phonetic CTC is enabled but the auxiliary phonetic_head is unavailable.')
        soft_features=_soft_semantic_decode(tts,logits,phonetic_temperature)
        head_dtype=next(phonetic_head.parameters()).dtype
        phonetic_logits=phonetic_head(soft_features.to(dtype=head_dtype)).float()
        input_lengths=torch.full(
            (phonetic_logits.shape[0],), phonetic_logits.shape[1], dtype=torch.long, device=text.device
        )
        target_lengths=torch.tensor([phonetic_target.numel()],dtype=torch.long,device=text.device)
        phonetic_loss=F.ctc_loss(
            F.log_softmax(phonetic_logits,dim=-1).transpose(0,1),
            phonetic_target.unsqueeze(0), input_lengths, target_lengths,
            blank=PHONETIC_BLANK, reduction='mean', zero_infinity=True,
        )
    if asr_model is not None and float(asr_weight)>0.0:
        if s2mel_prompt is None or ref_mel is None or style is None:
            raise RuntimeError('Differentiable ASR is enabled but S2Mel reference conditioning is missing.')
        asr_loss=_differentiable_asr_loss(
            tts,asr_model,logits,s2mel_prompt,ref_mel,style,target_text,
            language=asr_language,cfm_steps=asr_cfm_steps,temperature=asr_temperature,
        ).to(dtype=mel_loss.dtype)
    total=(
        mel_loss
        + max(0.0,float(semantic_weight))*text_loss
        + max(0.0,float(phonetic_weight))*phonetic_loss
        + max(0.0,float(asr_weight))*asr_loss
    )
    return total, mel_loss, text_loss, phonetic_loss, asr_loss


def _acoustic_sample_loss(tts, item, device):
    text, reference_text, spk, emo, codes, lang_id, phonetic_target, *_ = _prepare_sample(
        tts, item, device, include_auxiliary_conditioning=False,
    )
    with torch.autocast('cuda', dtype=torch.bfloat16):
        total, mel, *_ = _loss(tts, text, reference_text, spk, emo, codes, lang_id, phonetic_target)
    return total, mel


def _validation_loss(tts, rows, device, step, stop_event):
    adapter = tts.gpt.gpt
    was_training = adapter.training
    total = 0.0
    try:
        adapter.eval()
        with torch.no_grad():
            for item in rows:
                if stop_event.is_set():
                    return None
                try:
                    loss, _ = _acoustic_sample_loss(tts, item, device)
                    value = float(loss.float().item())
                    if not math.isfinite(value):
                        raise ValueError(f'Non-finite validation loss: {value}')
                except Exception as exc:
                    raise RuntimeError(f'Validation failed at step {step} for {item["audio"]}: {exc}') from exc
                total += value
    finally:
        adapter.train(was_training)
    return None if stop_event.is_set() else total / len(rows)


def _generate_eval(tts, config, step, out_dir, writer=None):
    if not config.get('eval_enabled') or not config.get('eval_text') or not config.get('eval_reference'):
        return None
    ref=Path(config['eval_reference'])
    if not ref.is_file(): return None
    try:
        # Rebuild the generation wrapper so it references the current PEFT-wrapped GPT.
        tts.gpt.post_init_gpt2_config(use_deepspeed=False,kv_cache=True,half=True)
        eval_dir=Path(out_dir)/'evaluation'; eval_dir.mkdir(parents=True,exist_ok=True)
        wav=eval_dir/f'eval-step-{int(step):06d}.wav'
        # Match the GUI's default inference profile so checkpoint previews
        # measure the same decoding behavior users get from Single Inference:
        # sampling enabled, top-p/top-k filtering, three beams, and the
        # default repetition/length controls.
        tts.infer(spk_audio_prompt=str(ref),text=config['eval_text'],output_path=str(wav),lang=str(config.get('eval_language','EN')),
                  reference_text=str(config.get('eval_reference_transcript') or ''), verbose=False,
                  max_text_tokens_per_segment=120,duration_factor=1.0,text_normalization=True,
                  do_sample=True,top_p=0.8,top_k=30,temperature=0.8,num_beams=3,
                  repetition_penalty=10.0,length_penalty=0.0,max_mel_tokens=1500)
        if writer is not None:
            try:
                import soundfile as sf
                data, sample_rate = sf.read(str(wav), dtype='float32', always_2d=True)
                audio = torch.from_numpy(data.T.copy())
                writer.add_audio('eval/audio', audio, int(step), sample_rate=int(sample_rate))
                writer.add_text('eval/text', str(config.get('eval_text', '')), int(step))
                writer.flush()
            except Exception as exc:
                print(f'[training][WARN] TensorBoard eval audio upload failed: {type(exc).__name__}: {exc}',flush=True)
        tts.gpt.gpt.train()
        return str(wav)
    except Exception as exc:
        print(f'[training] evaluation audio failed at step {step}: {type(exc).__name__}: {exc}',flush=True)
        try: tts.gpt.gpt.train()
        except Exception: pass
        return None


def _freeze_index_tts_modules(tts):
    """Freeze every nn.Module owned by the IndexTTS2 inference wrapper.

    IndexTTS2 is an orchestration wrapper and intentionally does not inherit
    from torch.nn.Module, so calling ``tts.parameters()`` is invalid. Its
    actual trainable graph lives in module-valued attributes such as ``gpt``,
    ``semantic_codec``, ``s2mel`` and ``bigvgan``.
    """
    seen = set()
    modules = []
    for value in vars(tts).values():
        if isinstance(value, torch.nn.Module) and id(value) not in seen:
            seen.add(id(value))
            modules.append(value)
    if not modules:
        raise RuntimeError("IndexTTS2 did not expose any torch modules to freeze.")
    for module in modules:
        module.requires_grad_(False)
    return modules


def train_lora(config, callback, stop_event):
    from peft import LoraConfig, get_peft_model, PeftModel
    from torch.utils.tensorboard import SummaryWriter
    from indextts.infer_v2_5 import IndexTTS2
    validation = config.get('validation')
    rows = _read_manifest(config['manifest'], validation.get('training_fingerprint') if validation else None)
    if not rows:
        raise RuntimeError('Prepared manifest is empty.')
    validation_rows = tuple(dict(row) for row in validation['rows']) if validation else ()
    if validation and not validation_rows:
        raise RuntimeError('Validation manifest is empty.')
    validation_interval = validation.get('interval_steps') if validation else None
    saved_config = dict(config)
    if validation:
        saved_config['validation'] = {key: value for key, value in validation.items() if key != 'rows'}
    if not torch.cuda.is_available():
        raise RuntimeError('IndexTTS 2.5 LoRA training requires an NVIDIA CUDA device.')
    seed=int(config.get('seed',1234)); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device='cuda:0'; model_dir=Path(config['model_dir']); out=Path(config['output_dir']); out.mkdir(parents=True,exist_ok=True)
    callback(message='Loading IndexTTS 2.5 training runtime…')
    if not (model_dir/'config.yaml').is_file():
        raise RuntimeError(f'IndexTTS 2.5 model config not found: {model_dir / "config.yaml"}')
    tts=IndexTTS2(cfg_path=str(model_dir/'config.yaml'),model_dir=str(model_dir),device=device,use_bf16=True,use_cuda_kernel=False,use_gpt_latent=False,use_torch_compile=False,use_deepspeed=False,use_accel=False,use_qwen_emo=False)
    # Use the stable native acoustic reconstruction objective. IndexTTS still
    # receives its target transcript as the normal TTS conditioning input, but
    # the optional auxiliary text-token, reference-transcript, phonetic-CTC,
    # CFM/S2Mel and Whisper objectives are intentionally disabled.
    callback(message='Using native acoustic LoRA objective; auxiliary text/ASR losses disabled.')
    # Only the autoregressive GPT transformer receives trainable LoRA parameters.
    # The remaining IndexTTS modules stay frozen.
    frozen_modules = _freeze_index_tts_modules(tts)
    callback(message=f"Frozen IndexTTS modules: {len(frozen_modules)}; injecting GPT LoRA adapters…")
    lcfg_kwargs=dict(
        r=int(config['rank']), lora_alpha=int(config['alpha']), lora_dropout=float(config['dropout']),
        bias='none', target_modules=['c_attn','c_proj','c_fc'],
    )
    lcfg=LoraConfig(**lcfg_kwargs)
    tts.gpt.gpt=get_peft_model(tts.gpt.gpt,lcfg)
    # IndexTTS constructs its inference modules in eval mode.  PEFT preserves
    # that state, which would silently disable LoRA dropout and leave the
    # adapter in inference mode during the actual optimization loop.
    tts.gpt.gpt.train()
    trainable=[p for p in tts.gpt.gpt.parameters() if p.requires_grad]
    if not trainable: raise RuntimeError('PEFT did not create trainable LoRA parameters.')
    trainable_count=sum(parameter.numel() for parameter in trainable)
    callback(message=f'LoRA trainable parameters: {trainable_count:,}')
    opt=torch.optim.AdamW(trainable,lr=float(config['lr']),betas=(0.9,0.95),eps=1e-8,weight_decay=0.01)
    resume=config.get('resume')
    start_step=0
    if resume and Path(resume).is_dir():
        callback(message=f'Resuming adapter from {resume}')
        loaded=PeftModel.from_pretrained(tts.gpt.gpt.get_base_model(),resume,is_trainable=True)
        tts.gpt.gpt=loaded
        tts.gpt.gpt.train()
        trainable=[p for p in loaded.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(f'Resumed adapter at {resume} has no trainable parameters.')
        callback(message=f'Resumed LoRA trainable parameters: {sum(parameter.numel() for parameter in trainable):,}')
        opt=torch.optim.AdamW(trainable,lr=float(config['lr']),betas=(0.9,0.95),eps=1e-8,weight_decay=0.01)
    micro=max(1,int(config['batch'])); accum=max(1,int(config['grad_accum'])); clip=float(config['grad_clip'])
    updates_per_epoch=max(1,math.ceil(len(rows)/(micro*accum)))
    total=int(config['steps']) if config['mode']=='Steps' else int(config['epochs'])*updates_per_epoch
    warm=max(20,min(100,int(round(total*.05))))
    min_lr_ratio=0.10
    def lr_lambda(current_step):
        if current_step < warm:
            return max(0.05,float(current_step+1)/float(warm))
        progress=min(1.0,max(0.0,(current_step-warm)/max(1,total-warm)))
        cosine=0.5*(1.0+math.cos(math.pi*progress))
        return min_lr_ratio+(1.0-min_lr_ratio)*cosine
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda)
    if resume and Path(resume).is_dir():
        state=Path(resume)/'trainer_state.pt'
        if state.is_file():
            st=torch.load(state,map_location='cpu'); start_step=int(st.get('step',0)); opt.load_state_dict(st['optimizer']); sched.load_state_dict(st['scheduler'])
    opt.zero_grad(set_to_none=True); started=time.time(); ema=None; step=start_step; cursor=(start_step*micro*accum)%len(rows)
    save_every=max(1,int(config['save_every']))

    def save_adapter(directory, saved_step):
        directory.mkdir(parents=True, exist_ok=True)
        tts.gpt.gpt.save_pretrained(str(directory), safe_serialization=True)
        torch.save({'step': saved_step, 'optimizer': opt.state_dict(),
                    'scheduler': sched.state_dict(), 'config': saved_config}, directory/'trainer_state.pt')

    # A resume from an earlier checkpoint can replay previously logged steps.
    # Retain the checkpoint's metrics and hide later events before appending.
    purge_step = start_step + 1 if validation_rows and start_step else None
    with SummaryWriter(str(out/'tensorboard'), purge_step=purge_step) as writer:
        if validation_rows:
            writer.add_custom_scalars({'Loss': {'Training and validation': [
                'Multiline', ['train/loss_ema', 'validation/loss'],
            ]}})
            writer.add_text('validation/dataset',
                            f'SHA-256: `{validation["fingerprint"]}`\n\nSamples: {len(validation_rows)}', start_step)
        while step<total and not stop_event.is_set():
            running_loss=0.0; running_mel=0.0; seen=0
            for ga in range(accum):
                for mb in range(micro):
                    if stop_event.is_set(): break
                    item=rows[cursor%len(rows)]; cursor+=1
                    sample_loss, mel_loss = _acoustic_sample_loss(tts, item, device)
                    if not torch.isfinite(sample_loss):
                        raise RuntimeError(f'Non-finite LoRA loss detected at step {step + 1}: {float(sample_loss.detach().float())}')
                    (sample_loss / (micro * accum)).backward()
                    running_loss+=float(sample_loss.detach().float().item())
                    running_mel+=float(mel_loss.detach().float().item())
                    seen+=1
                if stop_event.is_set(): break
            if not seen or stop_event.is_set(): break
            grad_norm=torch.nn.utils.clip_grad_norm_(trainable,clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f'Non-finite gradient norm detected at step {step + 1}: {float(grad_norm.detach().float())}')
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            raw=running_loss/seen; mel_raw=running_mel/seen
            ema=raw if ema is None else .95*ema+.05*raw; step+=1
            elapsed=time.time()-started; rate=(step-start_step)/max(elapsed,1e-6); eta=(total-step)/max(rate,1e-6)
            epoch=math.ceil(step/updates_per_epoch)
            current_lr=opt.param_groups[0]['lr']
            writer.add_scalar('train/loss_ema',ema,step); writer.add_scalar('train_raw/loss',raw,step); writer.add_scalar('train/lr',current_lr,step)
            writer.add_scalar('train/mel_loss',mel_raw,step)
            writer.add_scalar('train/grad_norm',float(grad_norm),step)
            writer.add_scalar('train/epoch',epoch,step)
            writer.flush()
            callback(step=step,total=total,epoch=epoch,loss=ema,elapsed=elapsed,eta=eta,message=f'Training step {step}/{total} | epoch {epoch} | lr {current_lr:.2e}')

            checkpoint = out / f'checkpoint-step-{step:06d}'
            checkpoint_due = step % save_every == 0 or step == total
            if checkpoint_due:
                save_adapter(checkpoint, step)

            if validation_rows and (step % (validation_interval or save_every) == 0 or step == total):
                callback(message=f'Running validation at step {step}…')
                try:
                    validation_loss = _validation_loss(tts, validation_rows, device, step, stop_event)
                except Exception as exc:
                    if not checkpoint_due:
                        try:
                            save_adapter(checkpoint, step)
                        except Exception as save_exc:
                            raise RuntimeError(f'Validation failed at step {step}; checkpoint save also failed: {save_exc}') from exc
                    raise
                if validation_loss is None:
                    break
                writer.add_scalar('validation/loss', validation_loss, step)
                writer.flush()
                callback(message=f'Validation loss at step {step}: {validation_loss:.6f}')

            if checkpoint_due:
                eval_wav=_generate_eval(tts,config,step,out,writer)
                writer.flush()
                callback(message=f'Saved checkpoint: {checkpoint.name}' + (f' | Eval: {Path(eval_wav).name}' if eval_wav else ''))
        final = out / 'final_adapter'
        save_adapter(final, step)
    del tts; torch.cuda.empty_cache()
    return {'step':step,'total':total,'output':str(final),'stopped':stop_event.is_set()}

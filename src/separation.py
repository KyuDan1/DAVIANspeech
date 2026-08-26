"""Voice / music separation backends.

Both backends expose the same contract: given a path to an audio file, return
``(voice, music)`` as float32 mono numpy arrays at 16 kHz.  That lets the
pipeline swap HTDemucs for SAM-Audio without touching anything downstream.
"""

from __future__ import annotations

import abc

import numpy as np
import torch
import torchaudio

TARGET_SR = 16_000
SILENCE_STD = 1e-8


def _to_mono_16k(waveform: torch.Tensor, source_sr: int) -> np.ndarray:
    """(C, T) tensor at source_sr -> (T,) float32 numpy at 16 kHz."""
    if waveform.dim() == 1:
        waveform = waveform[None]
    mono = waveform.mean(0, keepdim=True)
    if source_sr != TARGET_SR:
        mono = torchaudio.functional.resample(mono, source_sr, TARGET_SR)
    return mono[0].cpu().numpy().astype(np.float32)


class Separator(abc.ABC):
    """Splits a mixture into a voice stem and a music stem."""

    name: str

    @abc.abstractmethod
    def separate(self, audio_path) -> tuple[np.ndarray, np.ndarray]:
        """Return (voice_16k, music_16k) as mono float32 arrays."""


class HTDemucsSeparator(Separator):
    """Baseline backend: Demucs v4 'vocals' stem vs. the summed remainder."""

    name = "htdemucs"

    def __init__(self, device="cuda", repo=None, overlap=0.25, shifts=0):
        from demucs.pretrained import get_model

        # Demucs checkpoints predate torch 2.6's weights_only=True default.
        original_load = torch.load

        def trusting_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = trusting_load
        try:
            self.model = get_model("htdemucs", repo=repo)
        finally:
            torch.load = original_load

        self.model = self.model.cpu().eval()
        self.device = device
        self.overlap = overlap
        self.shifts = shifts

    def _load_track(self, audio_path) -> torch.Tensor:
        """Decode to (channels, T) at the model's rate.

        Demucs 4.1 dropped ``demucs.separate.load_track``; this follows what
        ``demucs.api.Separator`` now does -- sphn first, ffmpeg as the
        fallback for containers sphn will not open.
        """
        from demucs.audio import AudioFile, convert_audio

        try:
            import sphn

            data, source_sr = sphn.read(str(audio_path))
            return convert_audio(
                torch.from_numpy(data), int(source_sr),
                self.model.samplerate, self.model.audio_channels,
            ).float()
        except Exception:
            return AudioFile(audio_path).read(
                streams=0,
                samplerate=self.model.samplerate,
                channels=self.model.audio_channels,
            ).float()

    def separate(self, audio_path):
        from demucs.apply import apply_model

        waveform = self._load_track(audio_path)

        # Demucs expects roughly standardized input; a silent track would
        # divide by ~0, so short-circuit it.
        mono = waveform.mean(0)
        mean, std = mono.mean(), mono.std()
        if float(std) < SILENCE_STD:
            length = max(1, round(waveform.shape[-1] * TARGET_SR / self.model.samplerate))
            silence = np.zeros(length, dtype=np.float32)
            return silence, silence.copy()

        with torch.inference_mode():
            sources = apply_model(
                self.model,
                ((waveform - mean) / std)[None],
                device=self.device,
                shifts=self.shifts,
                split=True,
                overlap=self.overlap,
                progress=False,
            )[0]
        sources = sources * std + mean

        vocal_index = self.model.sources.index("vocals")
        voice = sources[vocal_index]
        music = torch.stack(
            [s for i, s in enumerate(sources) if i != vocal_index]
        ).sum(0)

        return (
            _to_mono_16k(voice, self.model.samplerate),
            _to_mono_16k(music, self.model.samplerate),
        )


class SamAudioSeparator(Separator):
    """SAM-Audio backend: text-prompted extraction of the voice stem.

    SAM-Audio returns a target stem plus its residual, so one prompted pass
    yields both halves -- the residual is by construction everything the voice
    prompt did not claim, which is the counterpart of Demucs' summed stems.
    """

    name = "sam-audio"

    DEFAULT_PROMPT = "a person speaking or singing"

    def __init__(self, device="cuda", checkpoint="facebook/sam-audio-large",
                 prompt=None, predict_spans=False):
        from sam_audio import SAMAudio, SAMAudioProcessor

        self.model = SAMAudio.from_pretrained(checkpoint).to(device).eval()
        self.processor = SAMAudioProcessor.from_pretrained(checkpoint)
        self.device = device
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.predict_spans = predict_spans

    def separate(self, audio_path):
        inputs = self.processor(
            audios=[str(audio_path)], descriptions=[self.prompt]
        ).to(self.device)
        with torch.inference_mode():
            result = self.model.separate(inputs, predict_spans=self.predict_spans)

        source_sr = self.processor.audio_sampling_rate
        return (
            _to_mono_16k(result.target[0].detach().cpu(), source_sr),
            _to_mono_16k(result.residual[0].detach().cpu(), source_sr),
        )


def build_separator(backend: str, device="cuda", **kwargs) -> Separator:
    if backend == "htdemucs":
        return HTDemucsSeparator(device=device, **kwargs)
    if backend == "sam-audio":
        return SamAudioSeparator(device=device, **kwargs)
    raise ValueError(f"Unknown separation backend: {backend}")

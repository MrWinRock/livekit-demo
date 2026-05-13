from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

import numpy as np
from livekit.agents import tts as lktts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid

logger = logging.getLogger("omnivoice_tts")

_DEFAULT_SAMPLE_RATE = 24000  # OmniVoice native sample rate

# Split on Thai/English sentence-ending punctuation to minimise first-chunk latency.
# Also splits on Thai ๆ-clause commas and the ellipsis so chunks stay short.
_SENTENCE_RE = re.compile(r"(?<=[.!?…。？！ฯ])\s*")


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
    return parts or [text]


class OmniVoiceTTS(lktts.TTS):
    def __init__(
        self,
        *,
        model_path: str,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        speed: Optional[float] = None,
        num_step: int = 8,
    ) -> None:
        super().__init__(
            capabilities=lktts.TTSCapabilities(streaming=False),
            sample_rate=_DEFAULT_SAMPLE_RATE,
            num_channels=1,
        )
        self._model_path = model_path
        self._language = language
        self._instruct = instruct
        self._ref_audio = ref_audio
        self._ref_text = ref_text
        self._speed = speed
        self._num_step = num_step
        self._ov_model = None
        self._voice_clone_prompt = None
        self._load_lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()
        return self._load_lock

    def _load_model_sync(self) -> None:
        import torch
        from omnivoice import OmniVoice

        use_cuda = torch.cuda.is_available()
        device = "cuda:0" if use_cuda else "cpu"
        dtype = torch.float16 if use_cuda else torch.float32
        logger.info(
            "Loading OmniVoice model: %s  device=%s  dtype=%s",
            self._model_path, device, dtype,
        )
        if not use_cuda:
            logger.warning(
                "CUDA not available — OmniVoice will run on CPU and be very slow. "
                "Ensure NVIDIA drivers and CUDA 12.8+ are installed."
            )

        self._ov_model = OmniVoice.from_pretrained(
            self._model_path,
            dtype=dtype,
            device_map=device,
        )
        self._sample_rate = self._ov_model.sampling_rate

        if self._ref_audio:
            logger.info("Creating voice clone prompt from: %s", self._ref_audio)
            self._voice_clone_prompt = self._ov_model.create_voice_clone_prompt(
                self._ref_audio,
                ref_text=self._ref_text,
            )
        logger.info("OmniVoice ready (sample_rate=%d  device=%s)", self._ov_model.sampling_rate, device)

    async def _ensure_loaded(self) -> None:
        if self._ov_model is not None:
            return
        async with self._get_lock():
            if self._ov_model is None:
                await asyncio.to_thread(self._load_model_sync)

    def prewarm(self) -> None:
        if self._ov_model is None:
            self._load_model_sync()

    @property
    def model(self) -> str:
        return "omnivoice"

    @property
    def provider(self) -> str:
        return "omnivoice"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "_OmniVoiceChunkedStream":
        return _OmniVoiceChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _OmniVoiceChunkedStream(lktts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: OmniVoiceTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._ov_tts = tts

    async def _run(self, output_emitter: lktts.AudioEmitter) -> None:
        ov = self._ov_tts
        await ov._ensure_loaded()

        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=ov._ov_model.sampling_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        sentences = _split_sentences(self._input_text)
        logger.debug("OmniVoice synthesizing %d chunk(s): %s", len(sentences), sentences)

        for sentence in sentences:
            def _generate(text: str = sentence) -> list:
                kwargs: dict = {"num_step": ov._num_step}
                if ov._voice_clone_prompt is not None:
                    kwargs["voice_clone_prompt"] = ov._voice_clone_prompt
                elif ov._instruct:
                    kwargs["instruct"] = ov._instruct
                if ov._language:
                    kwargs["language"] = ov._language
                if ov._speed is not None:
                    kwargs["speed"] = ov._speed
                return ov._ov_model.generate(text, **kwargs)

            import time
            t0 = time.perf_counter()
            audio_arrays = await asyncio.to_thread(_generate)
            elapsed = time.perf_counter() - t0

            total_samples = sum(len(a) for a in audio_arrays)
            audio_dur = total_samples / ov._ov_model.sampling_rate
            logger.info(
                "OmniVoice chunk: %.2fs audio in %.2fs (RTF %.2f)  steps=%d",
                audio_dur, elapsed, elapsed / max(audio_dur, 0.001), ov._num_step,
            )

            for arr in audio_arrays:
                pcm = np.clip(arr, -1.0, 1.0)
                output_emitter.push((pcm * 32767).astype(np.int16).tobytes())

            output_emitter.flush()

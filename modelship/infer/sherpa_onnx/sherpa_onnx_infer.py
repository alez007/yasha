import asyncio
import contextlib
import os
import platform
import threading
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np

from modelship.infer.base_infer import BaseInfer, ClientDisconnectedError
from modelship.infer.infer_config import ModelshipModelConfig, RawRequestProxy
from modelship.infer.model_resolver import ModelDownloadError
from modelship.infer.sherpa_onnx.bundle import resolve_bundle_dir
from modelship.infer.sherpa_onnx.registry import SherpaOnnxRegistryEntry
from modelship.logging import get_logger
from modelship.openai.protocol import ErrorResponse, RawSpeechResponse, SpeechRequest, create_error_response
from modelship.openai.utils.audio import sse_stream_speech
from modelship.utils import base_request_id
from modelship.utils.audio import to_pcm16, to_wav

logger = get_logger("infer.sherpa_onnx")


class SherpaOnnxInfer(BaseInfer):
    """In-process sherpa-onnx TTS loader. v1 scope: kokoro family only, CPU/CoreML
    only (no CUDA wheel wired)."""

    def __init__(self, model_config: ModelshipModelConfig):
        super().__init__(model_config)
        self.tts: Any = None
        self.entry: SherpaOnnxRegistryEntry | None = None

    def shutdown(self) -> None:
        self.tts = None

    def __del__(self):
        self.shutdown()

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self.tts, self.entry = await loop.run_in_executor(None, self._load)

    def _load(self) -> tuple[Any, SherpaOnnxRegistryEntry]:
        import sherpa_onnx

        assert self.model_config.model is not None
        try:
            bundle_dir, entry = resolve_bundle_dir(self.model_config.model)
        except Exception as e:
            raise ModelDownloadError(f"Failed to resolve sherpa_onnx bundle for '{self.model_config.name}': {e}") from e

        cfg = sherpa_onnx.OfflineTtsConfig()
        for slot, file in entry.files.items():
            setattr(cfg.model.kokoro, slot, os.path.join(bundle_dir, file.path))
        for slot, d in entry.dirs.items():
            setattr(cfg.model.kokoro, slot, os.path.join(bundle_dir, d.path))
        if entry.lexicon:
            cfg.model.kokoro.lexicon = ",".join(os.path.join(bundle_dir, f.path) for f in entry.lexicon)
        # Never take provider from config: an unsupported/typo'd value silently
        # falls back to CPU inside sherpa's C++ instead of raising.
        cfg.model.provider = "coreml" if platform.system() == "Darwin" else "cpu"
        cfg.model.num_threads = max(1, round(self.model_config.num_cpus))

        logger.info("loading sherpa_onnx model %r for '%s'", self.model_config.model, self.model_config.name)
        tts = sherpa_onnx.OfflineTts(cfg)
        if tts.num_speakers != len(entry.voice_names):
            raise ValueError(
                f"sherpa_onnx deployment '{self.model_config.name}': bundle reports {tts.num_speakers} speakers, "
                f"registry entry {self.model_config.model!r} expects {len(entry.voice_names)}. The bundle on disk "
                f"doesn't match this registry entry."
            )
        return tts, entry

    async def warmup(self) -> None:
        pass

    def _resolve_sid(self, voice: str) -> int | None:
        assert self.entry is not None
        if voice in self.entry.voice_names:
            return self.entry.voice_names.index(voice)
        if voice.isdigit():
            sid = int(voice)
            if 0 <= sid < len(self.entry.voice_names):
                return sid
        return None

    async def create_speech(
        self, request: SpeechRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | RawSpeechResponse | AsyncGenerator[str, None]:
        request_id = f"tts-{base_request_id(raw_request)}"
        assert self.entry is not None
        sid = self._resolve_sid(request.voice)
        if sid is None:
            return create_error_response(
                f"unknown voice {request.voice!r}. Valid voices: {', '.join(self.entry.voice_names)}"
            )

        if "stream_format" in request.model_fields_set and request.stream_format == "sse":
            logger.info("%s: voice=%s sid=%d stream=sse", request_id, request.voice, sid)
            return self.run_cancellable_stream(self._stream(request.input, sid, request.speed), raw_request)

        logger.info("%s: voice=%s sid=%d", request_id, request.voice, sid)
        try:
            wav = await self.run_cancellable(self._generate(request.input, sid, request.speed), raw_request)
        except ClientDisconnectedError:
            logger.info("%s aborted: client disconnected", request_id)
            return create_error_response("Client disconnected")
        return RawSpeechResponse(audio=wav)

    async def _generate(self, text: str, sid: int, speed: float) -> bytes:
        result = await asyncio.to_thread(self.tts.generate, text, sid, speed)
        samples = np.asarray(result.samples, dtype=np.float32)
        return to_wav(samples, result.sample_rate)

    async def _stream(self, text: str, sid: int, speed: float) -> AsyncGenerator[str, None]:
        async with contextlib.aclosing(self._generate_chunks(text, sid, speed)) as chunks:
            async for event in sse_stream_speech(chunks):
                yield event

    async def _generate_chunks(self, text: str, sid: int, speed: float) -> AsyncGenerator[tuple[bytes, int], None]:
        """callback fires once per sentence group (kokoro isn't autoregressive).
        Return 0 to abort, nonzero to continue — inverted from the bundled docstring."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        abort_event = threading.Event()
        sample_rate = self.tts.sample_rate
        _done = object()

        def on_chunk(samples: Any, _progress: float) -> int:
            pcm = to_pcm16(np.asarray(samples, dtype=np.float32))
            loop.call_soon_threadsafe(queue.put_nowait, pcm)
            return 0 if abort_event.is_set() else 1

        def run() -> None:
            try:
                self.tts.generate(text, sid, speed, callback=on_chunk)
            except BaseException as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _done)

        worker = asyncio.create_task(asyncio.to_thread(run))
        try:
            while True:
                item = await queue.get()
                if item is _done:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item, sample_rate
        finally:
            abort_event.set()
            await worker

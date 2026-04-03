"""
Phase 2: Speech transcription using faster-whisper tiny model.

Benchmarked on Apple Silicon M4:
- tiny model: 0.35s per 2s chunk (real-time capable)
- small model: 2.6s per 2s chunk (too slow)

VAD filter disabled — too aggressive for short 2s chunks.
beam_size=1 for maximum speed with negligible accuracy loss.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import time
import logging
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

TARGET_RATE = 16000


class Transcriber:
    """
    Wraps faster-whisper tiny for real-time speech transcription.

    Benchmarked at 0.35s per 2s audio chunk on Apple M4.
    Word timestamps enabled for temporal alignment with
    facial feature timeline.

    Usage:
        t = Transcriber()
        result = t.transcribe(audio_chunk)
        print(result['text'])    # transcript string
        print(result['words'])   # word-level timestamps
    """

    def __init__(self, model_size: str = "tiny"):
        logger.info(f"Loading faster-whisper {model_size}...")
        start = time.time()

        self._model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )

        elapsed = time.time() - start
        logger.info(f"faster-whisper {model_size} loaded in {elapsed:.1f}s")

        self._model_size           = model_size
        self._total_transcriptions = 0
        self._total_time           = 0.0

    def transcribe(
        self,
        audio_chunk: np.ndarray,
        language: str = "en"
    ) -> dict:
        """
        Transcribe a 2-second audio chunk.

        Args:
            audio_chunk: float32 numpy array at 16000Hz
                         Shape: (32000,) for 2 seconds
            language: Language code, "en" for English

        Returns:
            dict:
                text:               full transcript string
                words:              list of {word, start, end, probability}
                duration:           audio duration in seconds
                language:           str
                transcription_time: seconds taken
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return self._empty_result()

        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Skip if definitely silence (RMS below noise floor)
        # Threshold lowered to 0.0003 to catch quiet speech
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        if rms < 0.0003:
            return self._empty_result()

        # Normalize to 0.5 peak for consistent Whisper input
        max_val = np.abs(audio_chunk).max()
        if max_val > 0:
            audio_chunk = (audio_chunk / max_val * 0.5).astype(np.float32)

        start = time.time()

        try:
            segments, info = self._model.transcribe(
                audio_chunk,
                language=language,
                word_timestamps=True,
                vad_filter=False,           # Off — too aggressive for 2s
                beam_size=5,                # Better accuracy
                no_speech_threshold=0.6,    # Stricter — less noise transcribed as speech
                log_prob_threshold=-1.0,    # Stricter — drop low-confidence segments
                compression_ratio_threshold=2.4,
                condition_on_previous_text=False,  # Avoid hallucination chains
            )

            all_text  = []
            all_words = []

            for segment in segments:
                text = segment.text.strip()
                if text:
                    all_text.append(text)
                if segment.words:
                    for word in segment.words:
                        w = word.word.strip()
                        if w:
                            all_words.append({
                                "word":        w,
                                "start":       float(word.start),
                                "end":         float(word.end),
                                "probability": float(word.probability)
                            })

        except Exception as e:
            logger.warning(f"Transcription failed: {e}")
            return self._empty_result()

        elapsed = time.time() - start
        self._total_transcriptions += 1
        self._total_time           += elapsed

        return {
            "text":               " ".join(all_text).strip(),
            "words":              all_words,
            "duration":           len(audio_chunk) / TARGET_RATE,
            "language":           language,
            "transcription_time": elapsed
        }

    def get_stats(self) -> dict:
        avg = (
            self._total_time / self._total_transcriptions
            if self._total_transcriptions > 0 else 0
        )
        return {
            "model_size":            self._model_size,
            "total_transcriptions":   self._total_transcriptions,
            "avg_transcription_time": round(avg, 3)
        }

    def _empty_result(self) -> dict:
        return {
            "text":               "",
            "words":              [],
            "duration":           0.0,
            "language":           "en",
            "transcription_time": 0.0
        }

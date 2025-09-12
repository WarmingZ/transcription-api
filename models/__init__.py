"""
Модуль для локальних моделей транскрипції аудіо
"""

from .whisper_model import LocalWhisperModel
from .diarization import SimpleDiarizationService
from .transcription_service import LocalTranscriptionService

__all__ = [
    'LocalWhisperModel',
    'SimpleDiarizationService', 
    'LocalTranscriptionService'
]

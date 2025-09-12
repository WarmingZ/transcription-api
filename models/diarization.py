"""
Простий сервіс діаризації з чергуванням ролей Оператор/Клієнт
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple
import librosa
import webrtcvad

from .config import logger

class SimpleDiarizationService:
    """Простий сервіс діаризації з чергуванням ролей Оператор/Клієнт"""
    
    def __init__(self, transcription_service=None):
        self.vad = webrtcvad.Vad(2)  # Агресивність VAD (0-3, де 3 найагресивніша)
        self.transcription_service = transcription_service  # Посилання на основний сервіс для кешування
        
    def _detect_speech_segments(self, audio_path: str, min_silence_duration: float = 0.5) -> List[Tuple[float, float]]:
        """Виявляє сегменти мовлення за допомогою WebRTC VAD"""
        try:
            logger.info(f"Аналіз мовлення для {audio_path}...")
            
            # Завантажуємо аудіо як 16kHz mono PCM з кешуванням
            if self.transcription_service:
                logger.info(f"🔍 VAD аналіз файлу: {audio_path}")
                audio, sr = self.transcription_service._load_audio_cached(audio_path)
            else:
                logger.info(f"🔍 VAD аналіз файлу (без кешу): {audio_path}")
                audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Конвертуємо в int16 для WebRTC VAD
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Параметри для VAD
            frame_duration = 30  # мс
            frame_size = int(16000 * frame_duration / 1000)
            
            speech_segments = []
            current_segment_start = None
            in_speech = False
            
            # Обробляємо аудіо кадрами
            for i in range(0, len(audio_int16) - frame_size, frame_size):
                frame = audio_int16[i:i + frame_size]
                
                # Перевіряємо чи є мовлення в кадрі
                is_speech = self.vad.is_speech(frame.tobytes(), 16000)
                timestamp = i / 16000.0  # час в секундах
                
                if is_speech and not in_speech:
                    # Початок сегменту мовлення
                    current_segment_start = timestamp
                    in_speech = True
                elif not is_speech and in_speech:
                    # Кінець сегменту мовлення
                    segment_duration = timestamp - current_segment_start
                    if segment_duration >= 0.3:  # Мінімальна тривалість сегменту
                        speech_segments.append((current_segment_start, timestamp))
                    in_speech = False
            
            # Обробляємо останній сегмент
            if in_speech and current_segment_start is not None:
                speech_segments.append((current_segment_start, len(audio_int16) / 16000.0))
            
            logger.info(f"Знайдено {len(speech_segments)} сегментів мовлення")
            return speech_segments
            
        except Exception as e:
            logger.error(f"Помилка виявлення сегментів мовлення: {e}")
            return []
    
    def _merge_close_segments(self, segments: List[Tuple[float, float]], max_gap: float = 1.0) -> List[Tuple[float, float]]:
        """Об'єднує близькі сегменти мовлення"""
        if not segments:
            return []
        
        merged = [segments[0]]
        
        for current_start, current_end in segments[1:]:
            last_start, last_end = merged[-1]
            
            # Якщо сегменти близько один до одного, об'єднуємо їх
            if current_start - last_end <= max_gap:
                merged[-1] = (last_start, current_end)
            else:
                merged.append((current_start, current_end))
        
        return merged
    
    def assign_speakers(self, speech_segments: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """Призначає ролі Оператор/Клієнт чергуючи їх"""
        if not speech_segments:
            return []
        
        speaker_assignments = []
        
        for i, (start, end) in enumerate(speech_segments):
            # Чергуємо ролі: перший сегмент - Оператор, другий - Клієнт, і т.д.
            speaker = "Оператор" if i % 2 == 0 else "Клієнт"
            
            speaker_assignments.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "duration": end - start
            })
        
        return speaker_assignments
    
    def process_audio(self, audio_path: str) -> List[Dict[str, Any]]:
        """Повний процес діаризації аудіо"""
        try:
            logger.info("Початок простої діаризації...")
            
            # Виявляємо сегменти мовлення
            speech_segments = self._detect_speech_segments(audio_path)
            
            if not speech_segments:
                logger.warning("Сегменти мовлення не знайдено")
                return []
            
            # Об'єднуємо близькі сегменти
            merged_segments = self._merge_close_segments(speech_segments)
            
            # Призначаємо ролі
            speaker_assignments = self.assign_speakers(merged_segments)
            
            logger.info(f"Діаризація завершена: {len(speaker_assignments)} сегментів з ролями")
            return speaker_assignments
            
        except Exception as e:
            logger.error(f"Помилка діаризації: {e}")
            return []

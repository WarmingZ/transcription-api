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
        """Виявляє сегменти мовлення за допомогою WebRTC VAD з покращеною логікою"""
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
            
            # Покращені параметри для VAD
            frame_duration = 20  # мс (зменшено для кращої чутливості)
            frame_size = int(16000 * frame_duration / 1000)
            
            speech_segments = []
            current_segment_start = None
            in_speech = False
            silence_frames = 0
            min_silence_frames = int(min_silence_duration * 1000 / frame_duration)  # Кількість кадрів тиші
            
            logger.info(f"VAD параметри: frame_duration={frame_duration}ms, min_silence_frames={min_silence_frames}")
            
            # Обробляємо аудіо кадрами
            for i in range(0, len(audio_int16) - frame_size, frame_size):
                frame = audio_int16[i:i + frame_size]
                
                # Перевіряємо чи є мовлення в кадрі
                is_speech = self.vad.is_speech(frame.tobytes(), 16000)
                timestamp = i / 16000.0  # час в секундах
                
                if is_speech:
                    if not in_speech:
                        # Початок сегменту мовлення
                        current_segment_start = timestamp
                        in_speech = True
                        silence_frames = 0
                        logger.debug(f"Початок мовлення на {timestamp:.2f}с")
                    else:
                        silence_frames = 0  # Скидаємо лічильник тиші
                else:
                    if in_speech:
                        silence_frames += 1
                        # Кінець сегменту мовлення тільки після достатньої кількості кадрів тиші
                        if silence_frames >= min_silence_frames:
                            segment_duration = timestamp - current_segment_start
                            # Зменшено мінімальну тривалість сегменту для кращого виявлення коротких фраз
                            if segment_duration >= 0.1:  # Зменшено з 0.3 до 0.1 секунди
                                speech_segments.append((current_segment_start, timestamp))
                                logger.debug(f"Сегмент мовлення: {current_segment_start:.2f}с - {timestamp:.2f}с ({segment_duration:.2f}с)")
                            else:
                                logger.debug(f"Сегмент занадто короткий ({segment_duration:.2f}с), пропускаємо")
                            in_speech = False
                            silence_frames = 0
            
            # Обробляємо останній сегмент
            if in_speech and current_segment_start is not None:
                final_timestamp = len(audio_int16) / 16000.0
                segment_duration = final_timestamp - current_segment_start
                if segment_duration >= 0.1:  # Зменшено мінімальну тривалість
                    speech_segments.append((current_segment_start, final_timestamp))
                    logger.debug(f"Останній сегмент мовлення: {current_segment_start:.2f}с - {final_timestamp:.2f}с")
            
            logger.info(f"Знайдено {len(speech_segments)} сегментів мовлення")
            if speech_segments:
                logger.info(f"Перший сегмент: {speech_segments[0][0]:.2f}с - {speech_segments[0][1]:.2f}с")
                logger.info(f"Останній сегмент: {speech_segments[-1][0]:.2f}с - {speech_segments[-1][1]:.2f}с")
            
            return speech_segments
            
        except Exception as e:
            logger.error(f"Помилка виявлення сегментів мовлення: {e}")
            return []
    
    def _merge_close_segments(self, segments: List[Tuple[float, float]], max_gap: float = 1.0) -> List[Tuple[float, float]]:
        """Об'єднує близькі сегменти мовлення з покращеною обробкою початку"""
        if not segments:
            return []
        
        # Додаємо буфер на початок файлу для кращого виявлення перших слів
        processed_segments = []
        for start, end in segments:
            # Розширюємо початок сегменту на 0.2 секунди назад (але не менше 0)
            extended_start = max(0, start - 0.2)
            processed_segments.append((extended_start, end))
        
        merged = [processed_segments[0]]
        
        for current_start, current_end in processed_segments[1:]:
            last_start, last_end = merged[-1]
            
            # Якщо сегменти близько один до одного, об'єднуємо їх
            if current_start - last_end <= max_gap:
                merged[-1] = (last_start, current_end)
            else:
                merged.append((current_start, current_end))
        
        logger.info(f"Об'єднано {len(segments)} сегментів в {len(merged)} сегментів")
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

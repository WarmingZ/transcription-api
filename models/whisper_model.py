"""
Локальна модель faster-whisper для швидкої транскрипції аудіо
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import torch
import librosa
import soundfile as sf
import asyncio
from concurrent.futures import ProcessPoolExecutor
import time
import multiprocessing as mp

from .config import (
    MODELS_DIR, logger, SPEED_OPTIMIZED_BEAM_SIZE, SPEED_OPTIMIZED_VAD,
    SPEED_OPTIMIZED_CHUNK_SIZES, SUPPORTED_MODELS
)

# Імпорт soxr для швидкого ресемплінгу
try:
    import soxr
    SOXR_AVAILABLE = True
    logger.info("soxr доступний для швидкого ресемплінгу")
except ImportError:
    SOXR_AVAILABLE = False
    logger.warning("soxr недоступний, використовується librosa")

class LocalWhisperModel:
    """Локальна модель faster-whisper для швидкої транскрипції аудіо"""
    
    def __init__(self, model_size: str = "small", device: str = "cpu", transcription_service=None):
        self.model_size = model_size
        self.device = device
        self.model = None
        self.transcription_service = transcription_service  # Посилання на основний сервіс для кешування
        
    def load_model(self) -> bool:
        """Завантажує faster-whisper модель в локальну директорію проекту"""
        try:
            from faster_whisper import WhisperModel
            
            # Оптимізовані налаштування для максимальної швидкості
            if self.device == "cpu":
                # Для CPU: спробуємо int8_float16, fallback на int8
                try:
                    import psutil
                    memory_gb = psutil.virtual_memory().total / (1024**3)
                    if memory_gb >= 8:
                        compute_type = "int8_float16"  # Швидше ніж int8
                        logger.info("Спроба використати int8_float16 для CPU (швидше)")
                    else:
                        compute_type = "int8"  # Економніше по пам'яті
                        logger.info("Використовується int8 для CPU (економно)")
                except:
                    compute_type = "int8"
            else:
                # Для GPU: завжди float16
                compute_type = "float16"
                logger.info("Використовується float16 для GPU")
            
            # Шлях до моделі в локальній директорії проекту
            model_path = MODELS_DIR / f"faster-whisper-{self.model_size}"
            
            logger.info(f"Завантаження faster-whisper моделі {self.model_size} в {model_path}...")
            
            try:
                self.model = WhisperModel(
                    self.model_size, 
                    device=self.device, 
                    compute_type=compute_type,
                    download_root=str(MODELS_DIR)  # Завантажуємо в локальну директорію
                )
            except Exception as e:
                if "int8_float16" in str(e) and compute_type == "int8_float16":
                    # Fallback на int8 якщо int8_float16 не підтримується
                    logger.warning(f"int8_float16 не підтримується: {e}")
                    logger.info("Fallback на int8...")
                    compute_type = "int8"
                    self.model = WhisperModel(
                        self.model_size, 
                        device=self.device, 
                        compute_type=compute_type,
                        download_root=str(MODELS_DIR)
                    )
                else:
                    raise e
            
            logger.info(f"faster-whisper модель {self.model_size} завантажена успішно в {MODELS_DIR} (compute_type: {compute_type})")
            return True
            
        except Exception as e:
            logger.error(f"Помилка завантаження faster-whisper: {e}")
            return False
    
    def _get_optimal_model_size(self) -> str:
        """Визначає оптимальний розмір моделі на основі доступних ресурсів"""
        try:
            import psutil
            
            # Отримуємо інформацію про систему
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()
            
            # Визначаємо розмір моделі на основі ресурсів (тільки small та medium)
            if memory_gb >= 4 and cpu_count >= 2:
                return "medium"
            else:
                return "small"
        except Exception as e:
            logger.warning(f"Помилка визначення ресурсів: {e}, використовується small модель")
            return "small"
    
    def transcribe(self, audio_path: str, language: str = "uk") -> Dict[str, Any]:
        """Швидка транскрипція з faster-whisper (оптимізована для максимальної швидкості)"""
        if self.model is None:
            raise RuntimeError("Модель не завантажена")
        
        try:
            # Завантажуємо аудіо напряму як масив (без тимчасових файлів)
            audio, sr = self._load_and_preprocess_audio(audio_path)
            
            logger.info(f"Транскрипція аудіо масиву (тривалість: {len(audio)/sr:.1f}с)")
            
            # Використовуємо оптимізовані параметри для максимальної швидкості
            segments, info = self.model.transcribe(
                audio,  # Передаємо масив напряму
                language=language,
                beam_size=SPEED_OPTIMIZED_BEAM_SIZE,  # Завжди 1
                word_timestamps=True,
                vad_filter=SPEED_OPTIMIZED_VAD,  # Завжди False
                temperature=0.0,  # Детерміністичний результат
                best_of=1,  # Тільки один варіант
            )
            
            # Конвертуємо результат у формат, сумісний з оригінальним API
            segments_list = []
            full_text = ""
            
            for segment in segments:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                }
                segments_list.append(segment_dict)
                full_text += segment.text + " "
            
            result = {
                "text": full_text.strip(),
                "segments": segments_list,
                "duration": info.duration,
                "language": language
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Помилка транскрипції: {e}")
            raise
    
    def _load_and_preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Швидке завантаження та обробка аудіо з soxr (якщо доступний)"""
        try:
            # Завантажуємо аудіо файл
            if SOXR_AVAILABLE:
                # Використовуємо soxr для швидкого ресемплінгу
                # soxr.resample(audio, orig_sr, target_sr) - правильний API
                audio, orig_sr = librosa.load(audio_path, sr=None, mono=True)  # Спочатку завантажуємо
                if orig_sr != 16000:
                    audio = soxr.resample(audio, orig_sr, 16000)  # Потім ресемплімо через soxr
                    logger.debug(f"Завантажено через soxr: {orig_sr}Hz -> 16000Hz")
                else:
                    logger.debug(f"Аудіо вже 16kHz, ресемплінг не потрібен")
            else:
                # Fallback на librosa
                audio, sr = librosa.load(audio_path, sr=16000, mono=True)
                logger.debug(f"Завантажено через librosa: {sr}Hz -> 16000Hz")
            
            return audio, 16000
            
        except Exception as e:
            logger.warning(f"Помилка завантаження аудіо: {e}, використовується fallback")
            # Fallback на librosa
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            return audio, 16000
    
    def _convert_to_optimal_format(self, audio_path: str) -> str:
        """Оптимальна конвертація для українських аудіо 8kHz (телефонна якість) з GPU прискоренням"""
        try:
            logger.info(f"Конвертація файлу в оптимальний формат: {audio_path}")
            
            # Спробуємо використати GPU для швидшої обробки
            if torch.cuda.is_available():
                try:
                    audio, sr = self._convert_audio_gpu(audio_path)
                    logger.info("Використано GPU прискорення для конвертації")
                except Exception as e:
                    logger.warning(f"GPU конвертація не вдалася: {e}, використовується CPU")
                    audio, sr = librosa.load(audio_path, sr=None, mono=True)
            else:
                # Завантажуємо аудіо файл (завжди моно)
                audio, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Завжди конвертуємо в 16kHz для Whisper
            if sr != 16000:
                logger.info(f"Ресемплінг з {sr}Hz до 16000Hz")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            # Створюємо тимчасовий WAV файл (найкращий формат для Whisper)
            temp_wav = audio_path.rsplit('.', 1)[0] + '_temp.wav'
            
            # Зберігаємо як WAV PCM_16 mono (забороняємо MP3)
            sf.write(temp_wav, audio, 16000, format='WAV', subtype='PCM_16')
            
            logger.info(f"Файл конвертовано в оптимальний формат: {temp_wav}")
            return temp_wav
            
        except Exception as e:
            logger.warning(f"Помилка конвертації в оптимальний формат: {e}")
            return audio_path  # Повертаємо оригінальний файл
    
    def _convert_audio_gpu(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """GPU-прискорена конвертація аудіо"""
        try:
            import torchaudio
            
            # Завантажуємо аудіо на GPU
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Конвертуємо в моно якщо потрібно
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Ресемплінг на GPU якщо потрібно
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Конвертуємо назад в numpy
            audio = waveform.squeeze().cpu().numpy()
            
            return audio, sample_rate
            
        except ImportError:
            logger.warning("torchaudio не встановлений, використовується CPU")
            raise
        except Exception as e:
            logger.warning(f"Помилка GPU конвертації: {e}")
            raise
    
    def _post_process_transcription(self, result: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Мінімальна пост-обробка результату"""
        try:
            # Тільки базове очищення тексту
            if "text" in result:
                result["text"] = result["text"].strip()
            
            # Очищення сегментів
            if "segments" in result:
                for segment in result["segments"]:
                    if "text" in segment:
                        segment["text"] = segment["text"].strip()
            
            return result
            
        except Exception as e:
            logger.warning(f"Помилка пост-обробки: {e}")
            return result
    
    def _get_optimal_chunk_size(self, duration: float) -> int:
        """Визначає оптимальний розмір сегментів для максимальної швидкості"""
        if duration < 300:  # < 5 хв
            return SPEED_OPTIMIZED_CHUNK_SIZES['short']  # 20 сек
        elif duration < 1800:  # < 30 хв
            return SPEED_OPTIMIZED_CHUNK_SIZES['medium']  # 30 сек
        else:  # > 30 хв
            return SPEED_OPTIMIZED_CHUNK_SIZES['long']  # 45 сек
    
    def _split_audio_into_chunks(self, audio_path: str, chunk_duration: int = None) -> List[Tuple[np.ndarray, int]]:
        """Розбиває аудіо файл на сегменти як масиви (без тимчасових файлів)"""
        try:
            # Завантажуємо аудіо швидким методом
            audio, sr = self._load_and_preprocess_audio(audio_path)
            duration = len(audio) / sr
            
            # Визначаємо оптимальний розмір сегментів
            if chunk_duration is None:
                chunk_duration = self._get_optimal_chunk_size(duration)
            
            logger.info(f"Розбиття аудіо на сегменти по {chunk_duration} секунд (тривалість: {duration:.1f}с)...")
            
            # Якщо файл коротший за chunk_duration, повертаємо як є
            if duration <= chunk_duration:
                return [(audio, sr)]
            
            # Розбиваємо на сегменти як масиви
            chunk_arrays = []
            chunk_samples = chunk_duration * sr
            
            for i in range(0, len(audio), chunk_samples):
                chunk_audio = audio[i:i + chunk_samples]
                chunk_arrays.append((chunk_audio, sr))
            
            logger.info(f"Аудіо розбито на {len(chunk_arrays)} сегментів по {chunk_duration}с")
            return chunk_arrays
            
        except Exception as e:
            logger.warning(f"Помилка розбиття аудіо: {e}")
            # Fallback: завантажуємо весь файл
            audio, sr = self._load_and_preprocess_audio(audio_path)
            return [(audio, sr)]
    
    def _load_audio_cached(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Завантажує аудіо з диску (кешування вимкнено)"""
        # Кешування вимкнено - завжди завантажуємо з диску
        logger.debug(f"Завантаження аудіо з диску: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        return audio, sr
    
    def _transcribe_chunk(self, chunk_audio: np.ndarray, sr: int, language: str) -> Dict[str, Any]:
        """Транскрибує один сегмент аудіо (масив) з оптимізованими параметрами"""
        try:
            # Використовуємо оптимізовані параметри для максимальної швидкості
            segments, info = self.model.transcribe(
                chunk_audio,  # Передаємо масив напряму
                language=language,
                beam_size=SPEED_OPTIMIZED_BEAM_SIZE,  # Завжди 1
                word_timestamps=True,
                vad_filter=SPEED_OPTIMIZED_VAD,  # Завжди False
                temperature=0.0,  # Детерміністичний результат
                best_of=1,  # Тільки один варіант
            )
            
            segments_list = []
            full_text = ""
            
            for segment in segments:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                }
                segments_list.append(segment_dict)
                full_text += segment.text + " "
            
            return {
                "text": full_text.strip(),
                "segments": segments_list,
                "duration": info.duration,
                "language": language
            }
            
        except Exception as e:
            logger.error(f"Помилка транскрипції сегменту: {e}")
            return {
                "text": "",
                "segments": [],
                "duration": 0,
                "language": language
            }
    
    async def transcribe_parallel(self, audio_path: str, language: str = "uk", chunk_duration: int = None) -> Dict[str, Any]:
        """Паралельна транскрипція з ProcessPoolExecutor для максимальної швидкості"""
        if self.model is None:
            raise RuntimeError("Модель не завантажена")
        
        try:
            # Розбиваємо аудіо на сегменти як масиви
            chunk_arrays = self._split_audio_into_chunks(audio_path, chunk_duration)
            
            # Якщо тільки один сегмент, використовуємо звичайну транскрипцію
            if len(chunk_arrays) == 1:
                return self.transcribe(audio_path, language)
            
            # Динамічне визначення кількості процесів
            import os
            max_workers = min(os.cpu_count(), len(chunk_arrays), 8)  # Обмежуємо для стабільності
            logger.info(f"Паралельна транскрипція {len(chunk_arrays)} сегментів з {max_workers} процесами...")
            
            # Використовуємо ProcessPoolExecutor для CPU-bound завдань
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Створюємо завдання для кожного сегменту
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(executor, self._transcribe_chunk_worker, chunk_audio, sr, language)
                    for chunk_audio, sr in chunk_arrays
                ]
                
                # Чекаємо завершення всіх завдань
                chunk_results = await asyncio.gather(*tasks)
            
            # Об'єднуємо результати
            combined_text = ""
            combined_segments = []
            total_duration = 0
            time_offset = 0
            
            for i, result in enumerate(chunk_results):
                if result and result.get("text"):
                    combined_text += result["text"] + " "
                    total_duration += result.get("duration", 0)
                    
                    # Додаємо сегменти з корекцією часу
                    for segment in result.get("segments", []):
                        adjusted_segment = {
                            "start": segment["start"] + time_offset,
                            "end": segment["end"] + time_offset,
                            "text": segment["text"]
                        }
                        combined_segments.append(adjusted_segment)
                
                # Оновлюємо зміщення часу для наступного сегменту
                time_offset += result.get("duration", 0) if result else 0
            
            return {
                "text": combined_text.strip(),
                "segments": combined_segments,
                "duration": total_duration,
                "language": language
            }
            
        except Exception as e:
            logger.error(f"Помилка паралельної транскрипції: {e}")
            raise

    @staticmethod
    def _transcribe_chunk_worker(chunk_audio: np.ndarray, sr: int, language: str) -> Dict[str, Any]:
        """Worker функція для ProcessPoolExecutor (статичний метод)"""
        try:
            # Імпортуємо тут, щоб уникнути проблем з multiprocessing
            from faster_whisper import WhisperModel
            import torch
            
            # Визначаємо пристрій та compute_type
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                try:
                    import psutil
                    memory_gb = psutil.virtual_memory().total / (1024**3)
                    compute_type = "int8_float16" if memory_gb >= 8 else "int8"
                except:
                    compute_type = "int8"
            else:
                compute_type = "float16"
            
            # Завантажуємо модель в worker процесі
            model = WhisperModel("small", device=device, compute_type=compute_type)
            
            # Транскрибуємо з оптимізованими параметрами
            segments, info = model.transcribe(
                chunk_audio,
                language=language,
                beam_size=1,  # Максимальна швидкість
                word_timestamps=True,
                vad_filter=False,  # Вимкнено для швидкості
                temperature=0.0,
                best_of=1,
            )
            
            segments_list = []
            full_text = ""
            
            for segment in segments:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                }
                segments_list.append(segment_dict)
                full_text += segment.text + " "
            
            return {
                "text": full_text.strip(),
                "segments": segments_list,
                "duration": info.duration,
                "language": language
            }
            
        except Exception as e:
            logger.error(f"Помилка worker транскрипції: {e}")
            return {
                "text": "",
                "segments": [],
                "duration": 0,
                "language": language
            }

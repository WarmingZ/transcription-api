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
            
            # Оптимізовані налаштування для CPU (стабільний і швидкий варіант)
            if self.device == "cpu":
                compute_type = "int8"  # стабільний і швидкий варіант на CPU
                cpu_threads = min(4, os.cpu_count() or 4)
                logger.info(f"🚀 CPU оптимізація: compute_type=int8, cpu_threads={cpu_threads}")
            else:
                # Для GPU: завжди float16
                compute_type = "float16"
                cpu_threads = 1
                logger.info("Використовується float16 для GPU")
            
            # Шлях до моделі в локальній директорії проекту
            model_path = MODELS_DIR / f"faster-whisper-{self.model_size}"
            
            logger.info(f"Завантаження faster-whisper моделі {self.model_size} в {model_path}...")
            
            try:
                self.model = WhisperModel(
                    self.model_size, 
                    device=self.device, 
                    compute_type=compute_type,
                    cpu_threads=cpu_threads,
                    num_workers=1,
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
                        cpu_threads=cpu_threads,
                        num_workers=1,
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
            
            # Визначаємо розмір моделі на основі ресурсів (оптимізовано для сервера 8GB RAM + 4 CPU AMD)
            if torch.cuda.is_available():
                # Для GPU можна використовувати більші моделі
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    
                    if gpu_memory >= 8:  # 8GB+ GPU
                        logger.info(f"🚀 GPU {gpu_memory:.1f}GB + RAM {memory_gb:.1f}GB - оптимальна конфігурація для medium моделі")
                        return "medium"
                    elif gpu_memory >= 4:  # 4GB+ GPU
                        logger.info(f"🚀 GPU {gpu_memory:.1f}GB + RAM {memory_gb:.1f}GB - використовується small модель")
                        return "small"
                    else:
                        logger.info(f"🚀 GPU {gpu_memory:.1f}GB + RAM {memory_gb:.1f}GB - використовується base модель")
                        return "base"
                except:
                    logger.info(f"🚀 GPU доступна + RAM {memory_gb:.1f}GB - використовується small модель")
                    return "small"
            else:
                # Для CPU використовуємо швидшу модель
                if memory_gb >= 8 and cpu_count >= 4:
                    logger.info(f"🚀 Сервер {memory_gb:.1f}GB RAM + {cpu_count} CPU AMD - використовується base модель (швидше)")
                    return "base"
                elif memory_gb >= 6 and cpu_count >= 2:
                    logger.info(f"💾 Сервер {memory_gb:.1f}GB RAM + {cpu_count} CPU - використовується base модель")
                    return "base"
                else:
                    logger.info(f"💾 Сервер {memory_gb:.1f}GB RAM + {cpu_count} CPU - використовується base модель")
                    return "base"
        except Exception as e:
            logger.warning(f"Помилка визначення ресурсів: {e}, використовується base модель")
            return "base"
    
    def transcribe(self, audio_path: str, language: str = "uk") -> Dict[str, Any]:
        """Швидка транскрипція з faster-whisper (оптимізована для максимальної швидкості)"""
        if self.model is None:
            raise RuntimeError("Модель не завантажена")
        
        try:
            # Завантажуємо аудіо напряму як масив (без тимчасових файлів)
            audio, sr = self._load_and_preprocess_audio(audio_path)
            
            logger.info(f"Транскрипція аудіо масиву (тривалість: {len(audio)/sr:.1f}с)")
            logger.info(f"🔍 VAD фільтр: {'УВІМКНЕНО' if SPEED_OPTIMIZED_VAD else 'ВИМКНЕНО'}")
            
            # Використовуємо оптимізовані параметри для максимальної швидкості
            segments, info = self.model.transcribe(
                audio,  # Передаємо масив напряму
                language=language,
                beam_size=1,  # Мінімальний для швидкості
                word_timestamps=False,  # Вимкнено для швидкості
                vad_filter=True,  # УВІМКНЕНО для правильного виявлення початку мовлення
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # Мінімальна тривалість тиші
                    speech_pad_ms=200,  # Буфер навколо мовлення
                ),
                temperature=0.0,  # Детерміністичний результат
                best_of=1,  # Тільки один варіант
                condition_on_previous_text=False,  # Вимкнено для швидкості
                initial_prompt=None,  # Без попереднього тексту
                suppress_tokens=[-1],  # Придушення спеціальних токенів
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
        """Оптимальна конвертація для українських аудіо 8kHz (телефонна якість) з покращеною обробкою початку"""
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
                # Завантажуємо аудіо файл (завжди моно) з покращеними параметрами
                audio, sr = librosa.load(audio_path, sr=None, mono=True, offset=0.0)
            
            # Додаємо мінімальний буфер тиші на початок для кращого виявлення перших слів
            silence_buffer = np.zeros(int(0.05 * sr))  # 0.05 секунди тиші (зменшено)
            audio = np.concatenate([silence_buffer, audio])
            
            # Завжди конвертуємо в 16kHz для Whisper
            if sr != 16000:
                logger.info(f"Ресемплінг з {sr}Hz до 16000Hz")
                # Використовуємо кращий алгоритм ресемплінгу для збереження якості
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000, res_type='kaiser_best')
                sr = 16000
            
            # Створюємо тимчасовий WAV файл (найкращий формат для Whisper)
            temp_wav = audio_path.rsplit('.', 1)[0] + '_temp.wav'
            
            # Зберігаємо як WAV PCM_16 mono з покращеними параметрами
            sf.write(temp_wav, audio, 16000, format='WAV', subtype='PCM_16')
            
            logger.info(f"Файл конвертовано в оптимальний формат: {temp_wav} (додано буфер тиші на початок)")
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
        logger.info(f"🔍 Визначення розміру чанку для файлу тривалістю {duration:.1f}s")
        
        # Оптимізована логіка чанків для кращої продуктивності
        if duration < 60:  # < 1 хв - не розбиваємо
            chunk_size = int(duration) + 1
            logger.info(f"🔍 Файл < 1 хв: chunk_size = {chunk_size}s (не розбиваємо)")
            return chunk_size
        elif duration < 300:  # < 5 хв - малі чанки
            chunk_size = SPEED_OPTIMIZED_CHUNK_SIZES['short']
            logger.info(f"🔍 Файл < 5 хв: chunk_size = {chunk_size}s (short)")
            return chunk_size
        elif duration < 1800:  # < 30 хв - середні чанки
            chunk_size = SPEED_OPTIMIZED_CHUNK_SIZES['medium']
            logger.info(f"🔍 Файл < 30 хв: chunk_size = {chunk_size}s (medium)")
            return chunk_size
        else:  # > 30 хв - великі чанки
            chunk_size = SPEED_OPTIMIZED_CHUNK_SIZES['long']
            logger.info(f"🔍 Файл > 30 хв: chunk_size = {chunk_size}s (long)")
            return chunk_size
    
    def _split_audio_into_chunks(self, audio_path: str, chunk_duration: int = None) -> List[Tuple[np.ndarray, int, float, float]]:
        """Розбиває аудіо файл на сегменти як масиви з часовими мітками"""
        try:
            # Завантажуємо аудіо швидким методом
            audio, sr = self._load_and_preprocess_audio(audio_path)
            duration = len(audio) / sr
            
            # Визначаємо оптимальний розмір сегментів
            if chunk_duration is None:
                chunk_duration = self._get_optimal_chunk_size(duration)
            
            logger.info(f"Розбиття аудіо на сегменти по {chunk_duration} секунд (тривалість: {duration:.1f}с)...")
            logger.info(f"🔍 Діагностика: duration={duration:.1f}s, chunk_duration={chunk_duration}s")
            
            # Якщо файл коротший за chunk_duration, повертаємо як є
            if duration <= chunk_duration:
                logger.info(f"✅ Файл коротший за розмір чанку ({duration:.1f}с <= {chunk_duration}с), обробляємо як цілий")
                logger.info(f"✅ Це означає, що файл НЕ буде розбито на чанки - транскрипція почнеться з 0 секунди")
                return [(audio, sr, 0.0, duration)]
            
            # Розбиваємо на сегменти як масиви
            chunk_arrays = []
            chunk_samples = chunk_duration * sr
            
            logger.info(f"Розбиття на чанки: chunk_samples={chunk_samples}, total_samples={len(audio)}")
            
            for i in range(0, len(audio), chunk_samples):
                chunk_audio = audio[i:i + chunk_samples]
                chunk_start_time = i / sr
                chunk_end_time = (i + len(chunk_audio)) / sr
                
                # Детальна діагностика для першого чанку
                if len(chunk_arrays) == 0:
                    logger.info(f"🔍 ПЕРШИЙ ЧАНК: {chunk_start_time:.1f}s - {chunk_end_time:.1f}s ({len(chunk_audio)} зразків)")
                    logger.info(f"🔍 Початок аудіо: {chunk_start_time:.1f}s (має бути 0.0s!)")
                    if chunk_start_time > 0:
                        logger.warning(f"⚠️ УВАГА: Перший чанк починається з {chunk_start_time:.1f}s замість 0.0s!")
                
                logger.debug(f"Чанк {len(chunk_arrays)}: {chunk_start_time:.1f}s - {chunk_end_time:.1f}s ({len(chunk_audio)} зразків)")
                chunk_arrays.append((chunk_audio, sr, chunk_start_time, chunk_end_time))
            
            logger.info(f"🚀 Аудіо розбито на {len(chunk_arrays)} сегментів по {chunk_duration}с для швидкої паралельної обробки")
            logger.info(f"⚡ Це значно прискорить обробку файлу!")
            return chunk_arrays
            
        except Exception as e:
            logger.warning(f"Помилка розбиття аудіо: {e}")
            # Fallback: завантажуємо весь файл
            audio, sr = self._load_and_preprocess_audio(audio_path)
            return [(audio, sr, 0.0, len(audio) / sr)]
    
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
                vad_filter=True,  # УВІМКНЕНО для правильного виявлення початку мовлення
                vad_parameters=dict(
                    min_silence_duration_ms=300,  # Мінімальна тривалість тиші
                    speech_pad_ms=100,  # Буфер навколо мовлення
                ),
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
    
    async def transcribe_parallel(self, audio_path: str, language: str = "uk", chunk_duration: int = None, force_no_chunks: bool = False) -> Dict[str, Any]:
        """Паралельна транскрипція з ProcessPoolExecutor для максимальної швидкості"""
        if self.model is None:
            raise RuntimeError("Модель не завантажена")
        
        try:
            # Якщо вимкнено чанки, використовуємо звичайну транскрипцію
            if force_no_chunks:
                logger.info("🚫 Чанки вимкнено, використовується звичайна транскрипція")
                return self.transcribe(audio_path, language)
            
            # Розбиваємо аудіо на сегменти як масиви з часовими мітками
            chunk_data = self._split_audio_into_chunks(audio_path, chunk_duration)
            
            # Якщо тільки один сегмент, використовуємо звичайну транскрипцію
            if len(chunk_data) == 1:
                logger.info("✅ Тільки один чанк, використовується звичайна транскрипція")
                return self.transcribe(audio_path, language)
            
            # Оптимізоване визначення кількості процесів для сервера 8GB RAM + 4 CPU AMD
            import os
            cpu_count = os.cpu_count()
            max_workers = min(MAX_WORKERS, len(chunk_data), cpu_count)  # Оптимізовано для вашого сервера
            logger.info(f"🚀 Сервер {cpu_count} CPU AMD - паралельна транскрипція {len(chunk_data)} сегментів з {max_workers} процесами...")
            logger.info(f"⚡ Очікуване прискорення: ~{max_workers}x швидше ніж послідовна обробка")
            
            # Використовуємо ProcessPoolExecutor для CPU-bound завдань
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Створюємо завдання для кожного сегменту
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(executor, self._transcribe_chunk_worker, chunk_audio, sr, language)
                    for chunk_audio, sr, start_time, end_time in chunk_data
                ]
                
                # Чекаємо завершення всіх завдань
                chunk_results = await asyncio.gather(*tasks)
            
            # Об'єднуємо результати з правильними часовими зміщеннями
            combined_text = ""
            combined_segments = []
            total_duration = 0
            
            logger.info(f"Об'єднання {len(chunk_results)} чанків транскрипції...")
            
            for i, (result, chunk_info) in enumerate(zip(chunk_results, chunk_data)):
                chunk_audio, sr, chunk_start_time, chunk_end_time = chunk_info
                
                # ВИПРАВЛЕННЯ: Обробляємо ВСІ чанки, навіть якщо в них немає тексту
                if result:  # Тільки перевіряємо що результат існує
                    chunk_text = result.get("text", "").strip()
                    chunk_duration = result.get("duration", 0)
                    
                    logger.debug(f"Чанк {i}: реальний час {chunk_start_time:.1f}s-{chunk_end_time:.1f}s, текст='{chunk_text[:50]}...', duration={chunk_duration:.1f}s")
                    
                    # Додаємо текст (навіть якщо він порожній)
                    if chunk_text:
                        combined_text += chunk_text + " "
                    
                    # ВИПРАВЛЕННЯ: Використовуємо реальну тривалість чанку замість результату транскрипції
                    real_chunk_duration = chunk_end_time - chunk_start_time
                    total_duration += real_chunk_duration
                    
                    # Додаємо сегменти з корекцією часу на основі РЕАЛЬНОГО часу чанку
                    for segment in result.get("segments", []):
                        adjusted_segment = {
                            "start": segment["start"] + chunk_start_time,
                            "end": segment["end"] + chunk_start_time,
                            "text": segment["text"]
                        }
                        combined_segments.append(adjusted_segment)
                        logger.debug(f"Сегмент: {segment['start']:.1f}s-{segment['end']:.1f}s + {chunk_start_time:.1f}s = {adjusted_segment['start']:.1f}s-{adjusted_segment['end']:.1f}s")
                else:
                    # Якщо результат відсутній, все одно враховуємо тривалість чанку
                    real_chunk_duration = chunk_end_time - chunk_start_time
                    total_duration += real_chunk_duration
                    logger.warning(f"Чанк {i}: результат відсутній, але враховуємо тривалість {real_chunk_duration:.1f}s")
            
            # Діагностика результатів
            logger.info(f"📊 Статистика обробки:")
            logger.info(f"📊 Оброблено чанків: {len(chunk_results)}/{len(chunk_data)}")
            logger.info(f"📊 Знайдено сегментів: {len(combined_segments)}")
            logger.info(f"📊 Загальна тривалість: {total_duration:.1f}s")
            logger.info(f"📊 Довжина тексту: {len(combined_text)} символів")
            
            if combined_segments:
                logger.info(f"📊 Перший сегмент: {combined_segments[0]['start']:.1f}s - {combined_segments[0]['end']:.1f}s")
                logger.info(f"📊 Останній сегмент: {combined_segments[-1]['start']:.1f}s - {combined_segments[-1]['end']:.1f}s")
            else:
                logger.warning("⚠️ Не знайдено сегментів в об'єднаному результаті!")
            
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
            
            # Визначаємо пристрій та compute_type (оптимізовано для сервера 8GB RAM + 4 CPU AMD)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                try:
                    import psutil
                    memory_gb = psutil.virtual_memory().total / (1024**3)
                    cpu_count = psutil.cpu_count()
                    
                    if memory_gb >= 8 and cpu_count >= 4:
                        compute_type = "int8_float16"  # Оптимально для вашого сервера
                    elif memory_gb >= 8:
                        compute_type = "int8_float16"  # Швидше ніж int8
                    else:
                        compute_type = "int8"  # Економніше по пам'яті
                except:
                    compute_type = "int8"
            else:
                compute_type = "float16"
            
            # Завантажуємо модель в worker процесі
            model = WhisperModel("base", device=device, compute_type=compute_type)
            
            # Транскрибуємо з оптимізованими параметрами для швидкості
            segments, info = model.transcribe(
                chunk_audio,
                language=language,
                beam_size=1,  # Максимальна швидкість
                word_timestamps=False,  # Вимкнено для швидкості
                vad_filter=False,  # Вимкнено для швидкості
                temperature=0.0,
                best_of=1,
                condition_on_previous_text=False,  # Вимкнено для швидкості
                suppress_tokens=[-1],  # Придушення спеціальних токенів
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

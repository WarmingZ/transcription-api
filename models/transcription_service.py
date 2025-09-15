"""
Основний сервіс для локальної транскрипції аудіо з орфографічною корекцією
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import torch
import librosa
import soundfile as sf
import asyncio
import time

from .config import logger, LANGUAGE_TOOL_AVAILABLE, SPEED_OPTIMIZED_CHUNK_SIZES, SUPPORTED_MODELS, ENABLE_DIARIZATION, DIARIZATION_MAX_WORKERS
from .whisper_model import LocalWhisperModel
from .diarization import SimpleDiarizationService

class LocalTranscriptionService:
    """Сервіс для локальної транскрипції аудіо з орфографічною корекцією"""
    
    def __init__(self):
        self.models_loaded = False
        self.whisper_model = None
        self.diarization_service = None
        self.language_tool = None
        
        # Кешування тимчасово вимкнено
        
        # Ініціалізуємо LanguageTool для української мови (опціонально)
        if LANGUAGE_TOOL_AVAILABLE:
            try:
                import language_tool_python
                self.language_tool = language_tool_python.LanguageTool('uk-UA')
                logger.info("LanguageTool ініціалізовано для української мови")
            except Exception as e:
                logger.warning(f"LanguageTool недоступний (потрібен Java): {e}")
                self.language_tool = None
        else:
            logger.info("LanguageTool не встановлений - орфографічна корекція вимкнена")
        
    def load_models(self, model_size: str = "auto") -> bool:
        """Завантажує всі необхідні моделі"""
        try:
            logger.info("Завантаження локальних моделей...")
            
            # Визначаємо пристрій
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Використовується пристрій: {device}")
            
            # Визначаємо розмір моделі (оптимізовано для сервера 8GB RAM + 4 CPU AMD)
            if model_size == "auto":
                # Для української мови використовуємо звичайні моделі (distil тільки для англійської)
                try:
                    import psutil
                    memory_gb = psutil.virtual_memory().total / (1024**3)
                    cpu_count = psutil.cpu_count()
                    
                    if device == "cuda":
                        # Для GPU можна використовувати більші моделі
                        try:
                            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                            
                            if gpu_memory >= 8:  # 8GB+ GPU
                                model_size = "medium"  # Оптимально для GPU
                                logger.info(f"🚀 GPU {gpu_memory:.1f}GB + RAM {memory_gb:.1f}GB - використовується medium модель")
                            elif gpu_memory >= 4:  # 4GB+ GPU
                                model_size = "small"  # Безпечно для GPU
                                logger.info(f"🚀 GPU {gpu_memory:.1f}GB + RAM {memory_gb:.1f}GB - використовується small модель")
                            else:
                                model_size = "base"  # Для малих GPU
                                logger.info(f"🚀 GPU {gpu_memory:.1f}GB + RAM {memory_gb:.1f}GB - використовується base модель")
                        except:
                            model_size = "small"  # Fallback для GPU
                    else:
                        # Для CPU використовуємо small модель
                        if memory_gb >= 8 and cpu_count >= 4:
                            model_size = "small"  # Оптимально для стабільності на 8GB RAM
                            logger.info(f"🚀 Сервер {memory_gb:.1f}GB RAM + {cpu_count} CPU - використовується small модель (оптимально)")
                        elif memory_gb >= 6:
                            model_size = "small"  # Безпечно для 6GB+ RAM
                            logger.info(f"💾 Сервер {memory_gb:.1f}GB RAM - використовується small модель")
                        else:
                            model_size = "base"  # Для менших серверів
                            logger.info(f"💾 Сервер {memory_gb:.1f}GB RAM - використовується base модель")
                except:
                    model_size = "small"
            elif model_size not in SUPPORTED_MODELS:
                # Якщо вказано некоректний розмір, використовуємо small
                logger.warning(f"Невідомий розмір моделі: {model_size}, використовується small")
                model_size = "small"
            logger.info(f"Обрана модель для української мови: {model_size}")
            
            # Завантажуємо Whisper з посиланням на сервіс для кешування
            self.whisper_model = LocalWhisperModel(model_size=model_size, device=device, transcription_service=self)
            if not self.whisper_model.load_model():
                return False
            
            # Діаризація буде ініціалізована тільки при потребі (lazy loading)
            # self.diarization_service = SimpleDiarizationService(self)  # Видалено для оптимізації
            
            self.models_loaded = True
            logger.info("Всі моделі завантажені успішно")
            return True
            
        except Exception as e:
            logger.error(f"Помилка завантаження моделей: {e}")
            return False
    
    def _load_audio_cached(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Завантажує аудіо з диску (кешування вимкнено)"""
        # Кешування вимкнено - завжди завантажуємо з диску
        logger.debug(f"Завантаження аудіо з диску: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        return audio, sr
    
    def _correct_text(self, text: str, language: str) -> str:
        """Орфографічна корекція тексту для української мови"""
        if not text or language != "uk" or not self.language_tool:
            return text
        
        try:
            # Виправляємо помилки через LanguageTool
            matches = self.language_tool.check(text)
            import language_tool_python
            corrected_text = language_tool_python.utils.correct(text, matches)
            
            if corrected_text != text:
                logger.info("Застосовано орфографічну корекцію")
            
            return corrected_text
            
        except Exception as e:
            logger.warning(f"Помилка орфографічної корекції: {e}")
            return text
    
    def transcribe_simple(self, audio_path: str, language: str = "uk", use_parallel: bool = True, force_no_chunks: bool = True) -> Dict[str, Any]:
        """Швидка транскрипція з опціональним паралельним обробленням"""
        if not self.models_loaded:
            raise RuntimeError("Моделі не завантажені")
        
        start_time = time.time()
        try:
            logger.info("Початок швидкої транскрипції з faster-whisper...")
            
            # Використовуємо паралельну обробку для файлів довших за 2 хвилини (для швидкості)
            if use_parallel:
                # Перевіряємо тривалість файлу
                try:
                    import librosa
                    duration = librosa.get_duration(path=audio_path)
                    if duration > 60:  # Поріг 1 хвилина для швидкої обробки
                        logger.info(f"Файл довжиною {duration:.1f}с - використовується паралельна обробка")
                        import asyncio
                        try:
                            # Перевіряємо чи вже є запущений event loop
                            loop = asyncio.get_running_loop()
                            # Якщо є, використовуємо послідовну обробку (без попередження)
                            logger.info("Event loop активний, використовується послідовна обробка")
                            transcription_result = self.whisper_model.transcribe(audio_path, language)
                        except RuntimeError:
                            # Немає запущеного loop, створюємо новий
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                # Передаємо параметр для вимкнення чанків
                                transcription_result = loop.run_until_complete(
                                    self.whisper_model.transcribe_parallel(audio_path, language, None, force_no_chunks)
                                )
                            finally:
                                loop.close()
                    else:
                        logger.info(f"Файл короткий ({duration:.1f}с) - використовується послідовна обробка")
                        transcription_result = self.whisper_model.transcribe(audio_path, language)
                except Exception as e:
                    logger.warning(f"Помилка визначення тривалості: {e}, використовується послідовна обробка")
                    transcription_result = self.whisper_model.transcribe(audio_path, language)
            else:
                transcription_result = self.whisper_model.transcribe(audio_path, language)
            
            # Обробка результатів з орфографічною корекцією
            processed_result = self._process_simple_results(transcription_result, language)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Транскрипція завершена успішно за {elapsed_time:.2f} секунд")
            return processed_result
            
        except Exception as e:
            logger.error(f"Помилка транскрипції: {e}")
            raise
    
    def transcribe_with_diarization(self, audio_path: str, language: str = "uk", use_parallel: bool = True, force_no_chunks: bool = True) -> Dict[str, Any]:
        """Транскрипція з діаризацією (Оператор/Клієнт) з паралельною обробкою"""
        if not self.models_loaded:
            raise RuntimeError("Моделі не завантажені")
        
        # Lazy loading діаризації - ініціалізуємо тільки при потребі
        if not ENABLE_DIARIZATION:
            logger.warning("⚠️ Діаризація відключена в конфігурації")
            raise RuntimeError("Діаризація відключена в конфігурації")
        
        if self.diarization_service is None:
            logger.info("🔧 Ініціалізація діаризації (lazy loading)...")
            self.diarization_service = SimpleDiarizationService(self)
        
        start_time = time.time()
        try:
            logger.info("Початок транскрипції з діаризацією...")
            
            # Спочатку конвертуємо файл (якщо потрібно) та завантажуємо аудіо
            processed_audio_path = self.whisper_model._convert_to_optimal_format(audio_path)
            logger.info(f"🔄 Конвертований файл: {processed_audio_path}")
            logger.info("Завантаження аудіо для діаризації...")
            audio, sr = self._load_audio_cached(processed_audio_path)
            
            # Діагностика аудіо
            audio_duration = len(audio) / sr
            logger.info(f"📊 Аудіо діагностика: тривалість={audio_duration:.2f}с, частота={sr}Hz, зразків={len(audio)}")
            logger.info(f"📊 Перші 0.5с аудіо: min={audio[:int(0.5*sr)].min():.4f}, max={audio[:int(0.5*sr)].max():.4f}, rms={np.sqrt(np.mean(audio[:int(0.5*sr)]**2)):.4f}")
            
            # Робимо діаризацію з конвертованим файлом
            speaker_segments = self.diarization_service.process_audio(processed_audio_path)
            
            if not speaker_segments:
                logger.warning("Діаризація не знайшла сегменти, використовуємо просту транскрипцію")
                return self.transcribe_simple(audio_path, language, use_parallel)
            
            if use_parallel and len(speaker_segments) > 1:
                # Паралельна обробка сегментів з ProcessPoolExecutor
                logger.info(f"Паралельна обробка {len(speaker_segments)} сегментів діаризації...")
                processed_segments = self._process_diarization_segments_parallel(
                    audio, sr, speaker_segments, language
                )
            else:
                # Послідовна обробка (fallback)
                processed_segments = self._process_diarization_segments_sequential(
                    audio, sr, speaker_segments, language
                )
            
            # Підраховуємо статистику по дикторах
            speakers_stats = {}
            full_text = ""
            
            # Сортуємо сегменти за часом для правильного порядку
            processed_segments.sort(key=lambda x: x["start"])
            
            for segment in processed_segments:
                speaker = segment["speaker"]
                if speaker not in speakers_stats:
                    speakers_stats[speaker] = {
                        "speaker": speaker,
                        "segments_count": 0,
                        "total_duration": 0,
                        "first_segment": segment["start"],
                        "last_segment": segment["end"]
                    }
                
                speakers_stats[speaker]["segments_count"] += 1
                speakers_stats[speaker]["total_duration"] += segment["duration"]
                speakers_stats[speaker]["last_segment"] = max(speakers_stats[speaker]["last_segment"], segment["end"])
                
                # Додаємо час до тексту для кращого відстеження
                time_info = f"[{segment['start']:.1f}s-{segment['end']:.1f}s]"
                full_text += f"{time_info} [{speaker}]: {segment['text']}\n"
            
            result = {
                "text": full_text.strip(),
                "segments": processed_segments,
                "speakers": list(speakers_stats.values()),
                "duration": max([s["end"] for s in processed_segments]) if processed_segments else 0,
                "language": language,
                "diarization_type": "simple_alternating_parallel" if use_parallel else "simple_alternating"
            }
            
            elapsed_time = time.time() - start_time
            logger.info(f"Транскрипція з діаризацією завершена: {len(processed_segments)} сегментів за {elapsed_time:.2f} секунд")
            return result
            
        except Exception as e:
            logger.error(f"Помилка транскрипції з діаризацією: {e}")
            raise
    
    def _process_diarization_segments_parallel(self, audio: np.ndarray, sr: int, 
                                             speaker_segments: List[Dict[str, Any]], 
                                             language: str) -> List[Dict[str, Any]]:
        """Паралельна обробка сегментів діаризації з ProcessPoolExecutor"""
        try:
            import asyncio
            from concurrent.futures import ProcessPoolExecutor
            import os
            
            # Динамічне визначення кількості процесів (оптимізовано для сервера 8GB RAM + 4 CPU AMD)
            cpu_count = os.cpu_count()
            # Обмежуємо кількість процесів для діаризації (менше навантаження)
            max_workers = min(DIARIZATION_MAX_WORKERS, len(speaker_segments), cpu_count // 2)  # Тільки 50% CPU для діаризації
            logger.info(f"🚀 Сервер {cpu_count} CPU AMD - використовується {max_workers} процесів для паралельної діаризації (оптимізовано)")
            
            # Створюємо завдання для кожного сегменту
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                tasks = [
                    loop.run_in_executor(
                        executor, 
                        self._process_single_diarization_segment_worker,
                        audio, sr, segment, language
                    )
                    for segment in speaker_segments
                ]
                
                # Чекаємо завершення всіх завдань
                results = loop.run_until_complete(asyncio.gather(*tasks))
                loop.close()
            
            # Фільтруємо успішні результати
            processed_segments = [result for result in results if result is not None]
            
            logger.info(f"Паралельна обробка завершена: {len(processed_segments)}/{len(speaker_segments)} сегментів")
            return processed_segments
            
        except Exception as e:
            logger.error(f"Помилка паралельної обробки діаризації: {e}")
            # Fallback до послідовної обробки
            return self._process_diarization_segments_sequential(audio, sr, speaker_segments, language)
    
    @staticmethod
    def _process_single_diarization_segment_worker(audio: np.ndarray, sr: int,
                                                 speaker_info: Dict[str, Any], 
                                                 language: str) -> Optional[Dict[str, Any]]:
        """Worker функція для ProcessPoolExecutor (статичний метод)"""
        try:
            # Імпортуємо тут, щоб уникнути проблем з multiprocessing
            from faster_whisper import WhisperModel
            import torch
            
            start_time = speaker_info["start"]
            end_time = speaker_info["end"]
            speaker = speaker_info["speaker"]
            
            # Витягуємо сегмент аудіо з попередньо завантаженого масиву
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            
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
            model = WhisperModel("small", device=device, compute_type=compute_type)
            
            # Транскрибуємо сегмент з оптимізованими параметрами
            segments, info = model.transcribe(
                segment_audio,
                language=language,
                beam_size=1,  # Максимальна швидкість
                word_timestamps=True,
                vad_filter=True,  # УВІМКНЕНО для правильного виявлення початку мовлення
                vad_parameters=dict(
                    min_silence_duration_ms=300,  # Мінімальна тривалість тиші
                    speech_pad_ms=100,  # Буфер навколо мовлення
                ),
                temperature=0.0,
                best_of=1,
            )
            
            # Обробляємо результат
            segment_text = ""
            for segment in segments:
                segment_text += segment.text + " "
            
            if segment_text.strip():
                return {
                    "start": start_time,
                    "end": end_time,
                    "text": segment_text.strip(),
                    "speaker": speaker,
                    "duration": end_time - start_time
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Помилка worker обробки сегменту {speaker}: {e}")
            return None
    
    def _process_diarization_segments_sequential(self, audio: np.ndarray, sr: int,
                                               speaker_segments: List[Dict[str, Any]], 
                                               language: str) -> List[Dict[str, Any]]:
        """Послідовна обробка сегментів діаризації (fallback)"""
        processed_segments = []
        
        for speaker_info in speaker_segments:
            try:
                result = self._process_single_diarization_segment(audio, sr, speaker_info, language)
                if result:
                    processed_segments.append(result)
            except Exception as e:
                logger.warning(f"Помилка обробки сегменту {speaker_info.get('speaker', 'unknown')}: {e}")
                continue
        
        return processed_segments
    
    def _process_single_diarization_segment(self, audio: np.ndarray, sr: int,
                                          speaker_info: Dict[str, Any], 
                                          language: str) -> Optional[Dict[str, Any]]:
        """Обробка одного сегменту діаризації"""
        try:
            start_time = speaker_info["start"]
            end_time = speaker_info["end"]
            speaker = speaker_info["speaker"]
            
            # Витягуємо сегмент аудіо з попередньо завантаженого масиву
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Зберігаємо тимчасовий файл для сегменту
            segment_path = f"temp_segment_{start_time:.1f}_{end_time:.1f}.wav"
            sf.write(segment_path, segment_audio, sr, format='WAV', subtype='PCM_16')
            
            try:
                # Транскрибуємо сегмент з оптимізованими параметрами
                segment_duration = end_time - start_time
                if segment_duration < 10:  # Дуже короткі сегменти
                    beam_size = 1
                    vad_filter = False
                elif segment_duration < 30:
                    beam_size = 1
                    vad_filter = True
                else:
                    beam_size = 2
                    vad_filter = True
                
                # Використовуємо оптимізовані параметри
                segments, info = self.whisper_model.model.transcribe(
                    segment_path,
                    language=language,
                    beam_size=beam_size,
                    word_timestamps=True,
                    vad_filter=True,  # Завжди увімкнено для кращого виявлення
                    vad_parameters=dict(
                        min_silence_duration_ms=200,  # Зменшено для кращого виявлення коротких пауз
                        speech_pad_ms=150,  # Буфер навколо мовлення
                    ),
                )
                
                # Обробляємо результат
                segment_text = ""
                for segment in segments:
                    segment_text += segment.text + " "
                
                segment_result = {
                    "text": segment_text.strip(),
                    "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
                    "duration": info.duration,
                    "language": language
                }
                
                # Обробляємо результат
                if segment_result and segment_result.get("text"):
                    segment_text = segment_result["text"].strip()
                    
                    # Орфографічна корекція
                    if language == "uk":
                        segment_text = self._correct_text(segment_text, language)
                    
                    return {
                        "start": start_time,
                        "end": end_time,
                        "text": segment_text,
                        "speaker": speaker,
                        "duration": end_time - start_time
                    }
                
                return None
                
            finally:
                # Видаляємо тимчасовий файл
                try:
                    os.unlink(segment_path)
                except:
                    pass
                    
        except Exception as e:
            logger.warning(f"Помилка обробки сегменту {speaker}: {e}")
            return None
    
    def _process_simple_results(self, transcription_result: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Обробка результатів простої транскрипції з орфографічною корекцією"""
        try:
            if not transcription_result or not transcription_result.get("text"):
                return {
                    "text": "Текст не знайдено - аудіо може бути без мовлення або занадто тихим",
                    "segments": [],
                    "duration": 0,
                    "language": language
                }
            
            # Отримуємо текст
            text = transcription_result.get("text", "").strip()
            
            # Переконуємося, що текст правильно кодується в UTF-8
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
            
            # Орфографічна корекція для української мови
            if language == "uk":
                text = self._correct_text(text, language)
            
            # Отримуємо сегменти з орфографічною корекцією
            segments = []
            for segment in transcription_result.get("segments", []):
                segment_text = segment.get("text", "").strip()
                
                # Переконуємося, що текст правильно кодується в UTF-8
                if isinstance(segment_text, bytes):
                    segment_text = segment_text.decode('utf-8', errors='ignore')
                
                # Орфографічна корекція для кожного сегменту
                if language == "uk":
                    segment_text = self._correct_text(segment_text, language)
                
                segments.append({
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment_text
                })
            
            # Отримуємо тривалість
            duration = transcription_result.get("duration", 0)
            
            return {
                "text": text,
                "segments": segments,
                "duration": duration,
                "language": language
            }
            
        except Exception as e:
            logger.error(f"Помилка обробки результатів: {e}")
            return {
                "text": "Помилка обробки результатів транскрипції",
                "segments": [],
                "duration": 0,
                "language": language
            }

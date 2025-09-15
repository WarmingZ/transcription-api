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
import time

from .config import (
    MODELS_DIR, logger, SPEED_OPTIMIZED_BEAM_SIZE, SPEED_OPTIMIZED_VAD,
    SUPPORTED_MODELS, QUANTIZED_MODELS, CPU_COMPUTE_TYPE, GPU_COMPUTE_TYPE
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
            
            # Оптимізовані налаштування з quantized моделями (рекомендація ChatGPT)
            if self.device == "cpu":
                compute_type = CPU_COMPUTE_TYPE  # int8 для CPU
                cpu_threads = min(8, os.cpu_count() or 8)  # Використовуємо більше потоків для 8 CPU
                # Використовуємо quantized модель (compute_type="int8" автоматично quantized)
                model_name = self.model_size
                logger.info(f"🚀 CPU оптимізація: model={model_name} (quantized), compute_type={compute_type}, cpu_threads={cpu_threads}")
            else:
                # Для GPU: завжди float16
                compute_type = GPU_COMPUTE_TYPE
                cpu_threads = 1
                model_name = self.model_size
                logger.info(f"🚀 GPU оптимізація: model={model_name}, compute_type={compute_type}")
            
            # Шлях до моделі в локальній директорії проекту
            model_path = MODELS_DIR / f"faster-whisper-{self.model_size}"
            
            logger.info(f"Завантаження faster-whisper моделі {self.model_size} в {model_path}...")
            
            try:
                self.model = WhisperModel(
                    model_name,  # Використовуємо quantized модель якщо доступна
                    device=self.device, 
                    compute_type=compute_type,
                    cpu_threads=cpu_threads,
                    num_workers=2 if self.device == "cpu" else 1,  # Більше воркерів для CPU
                    download_root=str(MODELS_DIR)  # Завантажуємо в локальну директорію
                )
            except Exception as e:
                # Fallback: спробуємо з float16 якщо int8 не працює
                if self.device == "cpu" and compute_type == "int8":
                    logger.warning(f"Quantized модель (int8) не працює: {e}")
                    logger.info("Fallback на float16...")
                    self.model = WhisperModel(
                        self.model_size, 
                        device=self.device, 
                        compute_type="float16",
                        cpu_threads=cpu_threads,
                        num_workers=2 if self.device == "cpu" else 1,
                        download_root=str(MODELS_DIR)
                    )
                else:
                    raise e
            
            logger.info(f"faster-whisper модель {self.model_size} завантажена успішно в {MODELS_DIR} (compute_type: {compute_type})")
            return True
            
        except Exception as e:
            logger.error(f"Помилка завантаження faster-whisper: {e}")
            return False
    
    
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
    
    
    
    def _load_audio_cached(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Завантажує аудіо з диску (кешування вимкнено)"""
        # Кешування вимкнено - завжди завантажуємо з диску
        logger.debug(f"Завантаження аудіо з диску: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        return audio, sr
    
    


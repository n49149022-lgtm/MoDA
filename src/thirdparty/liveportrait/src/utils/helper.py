import os
import torch
import numpy as np
from omegaconf import OmegaConf

# Импорт базовых моделей LivePortrait/MoDA
# Пути указаны от корня репозитория (совместимо с sys.path.append в moda_test.py)
from src.thirdparty.liveportrait.src.models.appearance_feature_extractor import AppearanceFeatureExtractor
from src.thirdparty.liveportrait.src.models.motion_extractor import MotionExtractor
from src.thirdparty.liveportrait.src.models.spade_generator import SPADEGenerator
from src.thirdparty.liveportrait.src.models.warping_module import WarpingModule
from src.thirdparty.liveportrait.src.models.stitching_retargeting_module import StitchingRetargetingModule

MODEL_REGISTRY = {
    'appearance_feature_extractor': AppearanceFeatureExtractor,
    'motion_extractor': MotionExtractor,
    'spade_generator': SPADEGenerator,
    'warping_module': WarpingModule,
    'stitching_retargeting_module': StitchingRetargetingModule,
}

def load_model(model_config, model_type, device):
    """
    Загружает архитектуру модели и её чекпоинт.
    🔧 FIX: Отделяет 'model_path' от параметров инициализации, чтобы избежать TypeError.
            Корректно работает на CPU (map_location=device) и распаковывает вложенные state_dict.
    """
    print(f"[INFO] Initializing {model_type}...")

    if 'model_params' not in model_config:
        raise KeyError(f"Missing 'model_params' in config for {model_type}")

    params_key = f'{model_type}_params'
    if params_key not in model_config['model_params']:
        raise KeyError(f"Missing '{params_key}' in config['model_params']")

    # Копируем dict, чтобы не ломать оригинальный конфиг OmegaConf
    model_params = model_config['model_params'][params_key].copy()

    # 🔑 КЛЮЧЕВОЙ FIX: Извлекаем путь до передачи в __init__
    model_path = model_params.pop('model_path', None)

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")

    # Инициализация архитектуры чистыми параметрами
    model_class = MODEL_REGISTRY[model_type]
    model = model_class(**model_params).to(device)

    # Загрузка весов (если файл существует)
    if model_path and os.path.exists(model_path):
        print(f"⬇️ Loading checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        # Распаковка, если чекпоинт обёрнут в {'model': ...} или {'state_dict': ...}
        if isinstance(checkpoint, dict) and ('model' in checkpoint or 'state_dict' in checkpoint):
            state_dict = checkpoint.get('model', checkpoint.get('state_dict'))
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        print(f"✅ {model_type} weights loaded successfully.")
    elif model_path:
        print(f"⚠️ Checkpoint not found: {model_path}")

    model.eval()
    return model


# --- Вспомогательные утилиты (могут использоваться в других частях пайплайна) ---
def concat_feat(kp_source, kp_driving):
    """Конкатенация кпоинтов источника и драйвера."""
    return torch.cat([kp_source, kp_driving], dim=-1)

def calc_ratio(kp_source, kp_driving):
    """Расчёт масштабного коэффициента."""
    return kp_source / (kp_driving + 1e-8)

def is_video(path):
    """Проверка, является ли путь видеофайлом."""
    video_exts = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    return path.lower().endswith(video_exts)

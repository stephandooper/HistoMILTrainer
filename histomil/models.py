"""Extracted from CLAM and TRIDENT"""
from src.builder import create_model
from src._global_mappings import ENCODER_DIM_MAPPING


def import_model(model_name, pretrained_model, **kwargs):
    ENCODER_DIM_MAPPING.setdefault("mascaret", 1536)
    return create_model(f"{model_name}.base.{pretrained_model}", **kwargs)

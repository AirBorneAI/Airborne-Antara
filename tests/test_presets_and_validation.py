import pytest
from airborne_antara.presets import PRESETS, load_preset, list_presets, compare_presets
from airborne_antara.validation import ConfigValidator, validate_config

def test_list_presets():
    presets = list_presets()
    assert isinstance(presets, list)
    assert "production" in presets
    assert "fast" in presets
    assert "balanced" in presets

def test_load_preset():
    prod = load_preset("production")
    assert prod is not None
    assert prod.learning_rate > 0

def test_preset_merge():
    fast = PRESETS.fast()
    prod = PRESETS.production()
    merged = fast.merge(prod)
    assert merged is not None

def test_validate_config():
    valid, errors = validate_config(PRESETS.production())
    assert valid is True
    assert len(errors) == 0

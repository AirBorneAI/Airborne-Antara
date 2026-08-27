import pytest
import airborne_antara

def test_version():
    assert hasattr(airborne_antara, "__version__")
    assert airborne_antara.__version__ == "0.2.0"

def test_package_exports():
    expected_exports = [
        "AdaptiveFramework",
        "AdaptiveFrameworkConfig",
        "IntrospectionEngine",
        "PerformanceMonitor",
        "UnifiedMemoryHandler",
        "MetaController",
        "ProductionAdapter",
        "EnhancedConsciousnessCore",
        "SparseMoE",
        "WorldModel",
        "PerceptionGateway",
        "AdapterBank",
        "NeuralHealthMonitor",
        "PRESETS",
        "load_preset",
        "ConfigValidator",
        "KnowledgeGovernor",
    ]
    for export_name in expected_exports:
        obj = getattr(airborne_antara, export_name, None)
        assert obj is not None, f"Failed to export {export_name}"

def test_invalid_attribute():
    with pytest.raises(AttributeError):
        _ = airborne_antara.NonExistentAttribute123

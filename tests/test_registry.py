"""
Test the model registry functionality.
"""
import pytest
from whale_benchmark.utils.registry import register_model, get_model
from whale_benchmark.models.base import BaseModel


class TestModel1(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "test_model_1"
        self.kwargs = kwargs
    
    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)
    
    def fit(self, train_loader):
        return {"accuracy": 0.9}
    
    def predict(self, loader):
        return [0, 1, 0]


class TestModel2(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "test_model_2"
        self.kwargs = kwargs
    
    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)
    
    def fit(self, train_loader):
        return {"accuracy": 0.8}
    
    def predict(self, loader):
        return [1, 0, 1]


def test_register_model():
    """Test model registration and retrieval."""
    # Clear existing registry for this test
    from whale_benchmark.utils.registry import _MODEL_REGISTRY
    _MODEL_REGISTRY.clear()
    
    # Register the test models
    TestModel1 = register_model("test_model_1")(TestModel1)
    TestModel2 = register_model("test_model_2")(TestModel2)
    
    # Test getting models
    model1 = get_model("test_model_1", param1="value1")
    assert model1.name == "test_model_1"
    assert model1.kwargs["param1"] == "value1"
    
    model2 = get_model("test_model_2", param2="value2")
    assert model2.name == "test_model_2"
    assert model2.kwargs["param2"] == "value2"
    
    # Test getting a non-existent model
    with pytest.raises(ValueError):
        get_model("non_existent_model")


def test_decorator_registration():
    """Test that the decorator pattern works for model registration."""
    # Clear existing registry for this test
    from whale_benchmark.utils.registry import _MODEL_REGISTRY
    _MODEL_REGISTRY.clear()
    
    @register_model("decorated_model")
    class DecoratedModel(BaseModel):
        def __init__(self, **kwargs):
            super().__init__()
            self.name = "decorated_model"
            self.kwargs = kwargs
        
        @classmethod
        def from_config(cls, cfg):
            return cls(**cfg)
        
        def fit(self, train_loader):
            return {"accuracy": 0.95}
        
        def predict(self, loader):
            return [0, 0, 1]
    
    # Test that the model was registered
    model = get_model("decorated_model", param="value")
    assert model.name == "decorated_model"
    assert model.kwargs["param"] == "value" 
"""
Model registry for registering and retrieving models.
"""

# Dictionary to store registered models
_MODEL_REGISTRY = {}


def register_model(name):
    """
    Decorator for registering a model class.
    
    Args:
        name (str): Name to register the model under
        
    Returns:
        callable: Decorator function
    
    Example:
        @register_model("cnn")
        class CNNModel(BaseModel):
            pass
    """
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name, **kwargs):
    """
    Get a model class by name and instantiate it with the provided kwargs.
    
    Args:
        name (str): Name of the registered model
        **kwargs: Keyword arguments to pass to the model constructor
        
    Returns:
        BaseModel: Instantiated model
        
    Raises:
        ValueError: If the model name is not found in the registry
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found in registry. Available models: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models():
    """
    List all registered models.
    
    Returns:
        list: List of registered model names
    """
    return list(_MODEL_REGISTRY.keys())

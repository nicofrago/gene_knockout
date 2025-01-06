import torch
from models.simple_model import SimpleModel, SimpleModelV2

def get_models_dict():
    models = {
        'simple_model': SimpleModel,
        'simple_model_v2': SimpleModelV2
    }    
    return models
def model_factory(
        network_name:str, 
        num_classes:int, 
        input_size:int=512, 
        dropout_ratio:float=0.4
):
    models = get_models_dict()
    if network_name not in models:
        raise ValueError(f"Invalid network name: {network_name}")
    model_class = models[network_name]
    model = model_class(num_classes, input_size, dropout_ratio)
    return model

def load_model(model_name:str, num_classes:int, weights_path:str):
    model = model_factory(model_name, num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path))
    return model
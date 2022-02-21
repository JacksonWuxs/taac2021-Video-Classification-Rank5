from src.model.models.nextvlad_bert import NextVladBERT

def get_instance(name, paramters):
    model = {"NextVladBERT": NextVladBERT}[name]
    return model(paramters)
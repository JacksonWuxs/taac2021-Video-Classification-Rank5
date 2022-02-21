from src.model.fusion_head.fusion_se import SE

def get_instance(name, paramters):
    model = {'SE': SE}[name]
    return model(**paramters)
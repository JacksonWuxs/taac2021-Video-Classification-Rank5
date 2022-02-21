from src.model.classify_head.logistic_model import LogisticModel
from src.model.classify_head.moe_model import MoeModel

def get_instance(name, paramters_dict):
    model = {'LogisticModel': LogisticModel,
             'MoeModel': MoeModel}[name]
    return model(**paramters_dict)
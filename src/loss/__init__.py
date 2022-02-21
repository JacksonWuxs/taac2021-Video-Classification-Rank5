from src.loss.loss import CrossEntropyLoss
from src.loss.loss import SoftmaxLoss

def get_instance(name, paramters_dict):
    model = {'CrossEntropyLoss': CrossEntropyLoss,
            'SoftmaxLoss': SoftmaxLoss}[name]
    return model(**paramters_dict)
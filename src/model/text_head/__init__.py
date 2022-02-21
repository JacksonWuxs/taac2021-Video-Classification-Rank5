from src.model.text_head.bert_model import BERT

def get_instance(name, paramters):
    model = {'BERT': BERT}[name]
    return model(**paramters)
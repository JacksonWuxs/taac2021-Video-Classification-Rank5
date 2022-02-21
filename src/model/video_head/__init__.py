from src.model.video_head.nextvlad import NeXtVLAD

def get_instance(name, paramters_dict):
    model = {'NeXtVLAD': NeXtVLAD}[name]
    return model(**paramters_dict)
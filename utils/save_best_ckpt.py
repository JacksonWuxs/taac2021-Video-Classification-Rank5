import os
import sys


def iterate_files(folder, ftype=None):
    assert os.path.isdir(folder), "Path should be a folder"
    if isinstance(ftype, str):
        ftype = {ftype}
    elif isinstance(ftype, (list, tuple)):
        ftype = set(ftype)
        
    for file in os.listdir(folder):
        file = os.path.join(folder, file)
        if os.path.isfile(file) and \
           (ftype is None or os.path.split(file)[-1].split(".")[-1] in ftype):
            yield file
            continue
        elif os.path.isdir(file):
            for subfile in iterate_files(file, ftype):
                yield os.path.join(folder, subfile)
                
                
def remove_folder(folder):
    assert os.path.isdir(folder)
    for file in os.listdir(folder):
        file = os.path.join(folder, file)
        if os.path.isfile(file):
            os.remove(file)
            #print("Remove File: %s" % file)
            continue
        remove_folder(file)
    os.rmdir(folder)
    #print("Remove Foler: %s" % folder)


def select_best_model(folder):
    best_score, best_path = 0.0, None
    for model in os.listdir(folder):
        if model.startswith("step_"):
            _, step, score = model.split("_")
            score = float(score)
            if score > best_score:
                best_score, best_path = score, model
    return best_path


def remove_bad_model(folder):
    assert os.path.isdir(folder), "Path should be a folder"
    export = os.path.join(folder, "export")
    assert os.path.isdir(export), "Path should contains folder: export"
    
    best_path = select_best_model(export)
    #print("Best Model: %s" % best_path)
    best_step = int(best_path.split("_")[1])
    
    for file in os.listdir(folder):
        if file.startswith("model.ckpt-") and \
           int(file.split("-")[1].split(".")[0]) != best_step:
            os.remove(os.path.join(folder, file))
            #print("Remove: %s" % os.path.join(folder, file))
            
    for file in os.listdir(export):
        if file != best_path:
            remove_folder(os.path.join(export, file))
              
                
if __name__ == "__main__":
    remove_bad_model(sys.argv[1])

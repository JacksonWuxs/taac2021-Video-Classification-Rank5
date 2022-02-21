import numpy as np
import json
import sys


def load_label_dict(path):
    str2idx, idx2str = {}, []
    with open(path, encoding="utf8") as f:
        for line in f.readlines():
            label, index = line.strip().split('\t')
            str2idx[label] = int(index)
            idx2str.append(label)
    return str2idx, idx2str

        
def json_to_array(output, str2idx):
    target = np.zeros(82)
    output = output['result'][0]
    for label, score in zip(output['labels'],
                            output['scores']):
        target[str2idx[label]] = float(score)
    return target     


if __name__ == "__main__":
    num_folds = int(sys.argv[1])
    label_path = sys.argv[2] # './home/tione/notebook/VideoStructuring/dataset/label_id.txt'
    predict_path = sys.argv[3] # './home/tione/notebook/VideoStructuring/KFoldResults/test/results_{}/tagging_5k.json'
    output_path = sys.argv[4] # './home/tione/notebook/VideoStructuring/results/tagging_5k.json'
    topk = int(sys.argv[5]) # 20

    str2idx, idx2str = load_label_dict(label_path)
    
    full_predict = np.zeros((5000, num_folds, 82))
    for fold in range(num_folds):
        with open(predict_path.format(fold), encoding='utf8') as f:
            video_names = []
            for vid, (video_name, predict) in enumerate(sorted(json.load(f).items())):
                video_names.append(video_name)
                full_predict[vid, fold, :] = json_to_array(predict, str2idx)
            assert len(video_names) == 5000
    full_predict = full_predict.mean(axis=1)

    full_result = {}
    for video_name, scores in zip(video_names, full_predict):
        video_result = {"result": [{"labels": [], "scores":[]}]}
        for score, label in sorted(zip(scores, idx2str), reverse=True)[:topk]:
            video_result["result"][0]["labels"].append(label)
            video_result["result"][0]["scores"].append("%.4f" % score)
        full_result[video_name] = video_result

    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(full_result, f, ensure_ascii=False, indent=4)
            

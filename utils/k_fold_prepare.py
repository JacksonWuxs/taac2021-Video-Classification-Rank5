import os
import sys

from sklearn.model_selection import KFold


def parse_ground_truth(path):
    dataset = {}
    with open(path) as f:
        video_feat = []
        for row in f:
            row = row.strip()
            if len(row) == 0:
                assert len(video_feat) == 5
                dataset[os.path.split(video_feat[0])[-1].split(".")[0]] = video_feat
                video_feat = []
            else:
                video_feat.append(row)
    return dataset


if __name__ == "__main__":
    save_train, save_valid, config_path = sys.argv[3:6]    
    train_valid_data = parse_ground_truth(sys.argv[1])
    train_valid_data.update(parse_ground_truth(sys.argv[2]))
    videos = list(train_valid_data.keys())
    kf = KFold(n_splits=10, random_state=2021, shuffle=True)
    for i, (train_index, valid_index) in enumerate(kf.split(videos)):
        with open(config_path + 'config.tagging.5k.yaml', 'r') as fin, \
             open(config_path + 'config.tagging.5k.{}.yaml'.format(i), 'w') as fout:
            config = fin.read().replace("train.txt", 'train_{}.txt'.format(i))
            fout.write(config.replace("val.txt", 'valid_{}.txt'.format(i)))
        
        with open(save_train.format(i), 'w') as f:
            for idx in train_index:
                f.write(u"\n".join(train_valid_data[videos[idx]]) + '\n\n')
        
        with open(save_valid.format(i), 'w') as f:
            for idx in valid_index:
                f.write(u"\n".join(train_valid_data[videos[idx]]) + '\n\n')

#encoding: utf-8
import sys,os
sys.path.append(os.getcwd())
import glob
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import argparse
import time

import json
import utils.tokenization as tokenization
from utils.train_util import get_label_name_dict
from preprocess.multimodal_feature_extract import MultiModalFeatureExtract



def temporal_shift(src, shift_ratio=0.2):
    ts, fs = src.shape
    shift_dim = max(int(fs * shift_ratio) // 2, 1)
    out = np.zeros_like(src)
    out[1:, :shift_dim] = src[:-1, :shift_dim] # shift later
    out[:-1, -shift_dim:] = src[1:, -shift_dim:] # shift earlier
    out[:, shift_dim:-shift_dim] = src[:, shift_dim:-shift_dim] # no shift
    return out



class TaggingModel():
    def __init__(self, configs):
        tag_id_file = configs.get('tag_id_file', None)
        model_pb = configs.get('model_pb', None)
        
        if tag_id_file is None:
            raise
        if model_pb is None:
            raise
        
        self.label_name_dict = get_label_name_dict(tag_id_file, None)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        meta_graph_def = tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], model_pb)
        signature_def = meta_graph_def.signature_def
        self.signature = signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        batch_size = configs.get('video_feats_extractor_batch_size', 128)
        self.jieba_cut = self.load_jieba()
        self.zh_w2v = self.load_w2v("pretrained/word_embed/Tencent_AILab_ChineseEmbedding_cut100w.txt")
        self.tokenizer = tokenization.FullTokenizer(vocab_file='pretrained/bert/chinese_L-12_H-768_A-12/vocab.txt')
        imgfeat_extractor = configs.get('imgfeat_extractor', 'efficientnet')
        self.feat_extractor = MultiModalFeatureExtract(batch_size=batch_size, imgfeat_extractor= imgfeat_extractor, 
                             extract_video = True, extract_audio = True, extract_text = False)

    def image_preprocess(self, image, rescale=224):
        #resize to 224 and normlize to 0-1, then perform f(x)= 2*(x-0.5)
        if isinstance(image, type(None)):
            return np.zeros((rescale, rescale, 3))
        if image.shape[0] !=rescale:
            image = cv2.resize(image, (rescale, rescale))
        return 2*(image/(np.max(image)+1e-10) - 0.5)
    
    def load_w2v(self, path):
        embeddings = {}
        with open(path, encoding="utf8") as f:
            for row in f:
                char, score = row.strip().split(" ", 1)
                embeddings[char] = np.fromstring(score, sep=" ")
        return embeddings

    def load_jieba(self):
        import jieba
        return lambda sentence: list(jieba.cut(sentence))
    
    def text_preprocess(self, data, max_len=128):
        text = data["video_ocr"] + data["video_asr"]
        tokens = ['[CLS]'] + self.tokenizer.tokenize(text.replace("|", ""))
        ids = self.tokenizer.convert_tokens_to_ids(tokens)[:max_len]
        return ids + [0]*(max_len-len(ids))

    def video_preprocess(self, video, data, max_frame_num):
        tokens = self.jieba_cut(data["video_asr"].strip().replace("|", ""))
        if len(tokens) < len(video):
            text = data["video_ocr"]
            for char in '''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*|()~`·、[]【】？、。，,.';；‘:""''':
                text = text.replace(char, "")
            tokens = self.jieba_cut(text)
        window_size = len(tokens) // len(video)
        embeddings = []
        for i in range(len(video)):
            start = i * window_size
            frame_embed = np.zeros((200,))
            k = 0.0
            for token in tokens[start: start+window_size]:
                if token in self.zh_w2v:
                    frame_embed += self.zh_w2v[token]
                    k += 1.0
            if k > 0:
                frame_embed = frame_embed / k
            embeddings.append(frame_embed)
        embeddings = np.vstack(embeddings)
        video = np.hstack([video, embeddings])
        video = temporal_shift(video)[:max_frame_num]
        return np.vstack([video, np.zeros((max_frame_num - len(video), video.shape[-1]))])

    def preprocess(self, feat_dict, max_frame_num=300):
        ret_dict = {}
        for name, feats in feat_dict.items():
            if name == 'video':
                feats = self.video_preprocess(feats, feat_dict["text"], max_frame_num)
            elif name == 'audio':
                feats = np.zeros((max_frame_num, len(feat_dict['audio'][0])))
                valid_num = min(max_frame_num, len(feat_dict['audio']))
                feats[:valid_num] = feat_dict['audio']
            elif name == 'text':
                feats = self.text_preprocess(feat_dict['text'], 300)
            elif name == 'image':
                feats = self.image_preprocess(feats)
            else:
                continue
            ret_dict[name] = feats
        return ret_dict

    def load_multimodal_feat(self, test_file, feat_dir):
        assert os.path.exists(feat_dir)
        feat_dict = {}
        # load video feat
        video_id = os.path.basename(test_file).split('.m')[0]
        feat_dict['video'] = np.load(os.path.join(feat_dir,'video_npy' ,'Youtube8M', 'tagging', video_id + '.npy'))
        try:
            feat_dict['audio'] = np.load(os.path.join(feat_dir, 'audio_npy', 'Vggish', 'tagging', video_id + '.npy'))
        except:
            print("Loading Audio Feature for video: %s is failed!" % video_id)
            feat_dict['audio'] = np.zeros((10, 128))
        feat_dict['image'] = cv2.imread(os.path.join(feat_dir, 'image_jpg','tagging', video_id + '.jpg'), 1)
        feat_dict['text'] = eval(open(os.path.join(feat_dir, 'text_txt', 'tagging', video_id + '.txt')).read().strip())
        return feat_dict

    def inference(self, test_file, load_feat=False, feat_dir=None):
        with self.sess.as_default() as sess:
            feat_dict = self.load_multimodal_feat(test_file, feat_dir)

            if 'text' not in feat_dict:
                feat_dict['text'] = ""
            feat_dict_preprocess = self.preprocess(feat_dict)
            
            feed_dict ={}
            # Get input tensor.
            for key in feat_dict:
                if key in self.signature.inputs:
                    feed_dict[self.signature.inputs[key].name] = [feat_dict_preprocess[key]]
                
            if 'video_frames_num' in self.signature.inputs:
                feed_dict[self.signature.inputs['video_frames_num'].name] = [len(feat_dict['video'])]
            if 'audio_frames_num' in self.signature.inputs:
                feed_dict[self.signature.inputs['audio_frames_num'].name] = [len(feat_dict['audio'])]
                
            # Get output tensor.
            class_indexes = self.signature.outputs['class_indexes'].name
            predictions = self.signature.outputs['predictions'].name
            #video_embedding = self.signature.outputs['video_embedding'].name #(Optional)
            
            start_time = time.time()
            class_indexes,predictions = sess.run([class_indexes,predictions], feed_dict)
            end_time = time.time()
            
            print("multi-modal tagging model forward cost time: {} sec".format(end_time -start_time))


            labels=[self.label_name_dict[index] for index in class_indexes[0]]
            scores = predictions[0]

        return labels, scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pb', default='checkpoints/ds/export/step_10000_0.6879',type=str)
    parser.add_argument('--tag_id_file', default='dataset/dict/tag-id-deconstruction_b0.txt')
    parser.add_argument('--test_dir', default='../SceneSeg/data/ad300/video')
    parser.add_argument('--postfix', default='.mp4', type=str, help='test file type')
    parser.add_argument('--load_feat', type=int, default=1) # 加载预训练的特征
    parser.add_argument('--feat_dir', type=str, default=None) # 视频特征目录, 不同按照固定目录格式存放
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--output', default="results/result_for_vis.txt", type=str) #用于可视化文件
    parser.add_argument('--output_json', default="results/outjson.txt", type=str) #用于模型精度评估
    args = parser.parse_args()
    
    configs={'tag_id_file': args.tag_id_file, 'model_pb': args.model_pb}
    model = TaggingModel(configs)
    test_files = glob.glob(args.test_dir+'/*'+args.postfix)
    test_files.sort()    
    output_result = {}
    
    #clean temp file
    if os.path.exists(args.output):
        os.remove(args.output)

    for test_file in test_files:
        labels, scores = model.inference(test_file, args.load_feat, args.feat_dir)
        if args.output_json is not None:
            cur_output = {}
            output_result[test_file.split("/")[-1]] = cur_output
            cur_output["result"] = [{"labels": labels[:args.top_k], "scores": ["%.8f" % scores[i] for i in range(args.top_k)]}]
            print(cur_output)

        if args.output is not None:
            with open(args.output, 'a+') as f:
                video_id = test_file.split('/')[-1].split('.')[0]
                scores = [scores[i] for i in range(args.top_k)]
                f.write("{}\t{}\n".format(video_id, "\t".join(["{}##{:.3f}".format(labels[i], scores[i]) for i in range(len(scores))])))
        print("-"*100)
    
    with open(args.output_json, 'w', encoding="utf-8") as f:
        json.dump(output_result, f, ensure_ascii=False, indent = 4)

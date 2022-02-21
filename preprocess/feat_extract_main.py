#encoding: utf-8
import sys,os
sys.path.append(os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import argparse
import tqdm
import random
import glob
import traceback

from multimodal_feature_extract import MultiModalFeatureExtract


def process_file(file_path, frame_npy_path, audio_npy_path, text_txt_path, image_jpg_path):
    if not os.path.exists(file_path):
        return
    try:
        print(file_path)
        gen.extract_feat(file_path, frame_npy_path, audio_npy_path, text_txt_path, image_jpg_path)
    except Exception as e:
        print(traceback.format_exc())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_files_dir', default=None,type=str)
    parser.add_argument('--postfix', default='mp4', type=str)
    parser.add_argument('--frame_npy_folder', default='dataset/frame_npy', type=str)
    parser.add_argument('--audio_npy_folder', default='dataset/audio_npy', type=str)
    parser.add_argument('--image_jpg_folder', default='dataset/image_jpg', type=str)
    parser.add_argument('--text_txt_folder', default='dataset/text_txt', type=str)
    parser.add_argument('--datafile_path', default='dataset/datafile.txt')
    
    parser.add_argument('--extract_type', default=0, type=int) #0:ALL #1:VIDEO #2: AUDIO #3: TEXT
    
    parser.add_argument('--image_batch_size', default=32, type=int)
    parser.add_argument('--imgfeat_extractor', default='Youtube8M', type=str)
    parser.add_argument('--do_logging', default=0, type=int)
    
    args = parser.parse_args()
    os.makedirs(args.frame_npy_folder, exist_ok=True)
    os.makedirs(args.audio_npy_folder, exist_ok=True)
    os.makedirs(args.text_txt_folder, exist_ok=True)
    os.makedirs(args.image_jpg_folder, exist_ok=True)
    gen =  MultiModalFeatureExtract(batch_size = args.image_batch_size,
                             imgfeat_extractor = args.imgfeat_extractor,
                             extract_video = args.extract_type==0 or args.extract_type==1,
                             extract_audio = args.extract_type==0 or args.extract_type==2,
                             extract_text = args.extract_type==0 or args.extract_type==3)
            
    file_paths = glob.glob(args.test_files_dir+'/*.'+args.postfix)
    random.shuffle(file_paths)
    files = tqdm.tqdm(file_paths, total=len(file_paths)) if args.do_logging == 1 else file_paths
    for file_path in files:
        vid = os.path.basename(file_path).split('.m')[0]
        frame_npy_path = os.path.join(args.frame_npy_folder, vid+'.npy')
        audio_npy_path = os.path.join(args.audio_npy_folder, vid+'.npy')
        image_jpg_path = os.path.join(args.image_jpg_folder, vid+'.jpg')
        text_txt_path = os.path.join(args.text_txt_folder, vid+'.txt')
        if args.extract_type == 1:
            audio_npy_path, text_txt_path, image_jpg_path = None, None, None
        elif args.extract_type == 2:
            frame_npy_path, text_txt_path, image_jpg_path = None, None, None
        elif args.extract_type == 3:
            frame_npy_path, audio_npy_path, image_jpg_path = None, None, None
        elif args.extract_type ==4:
            frame_npy_path, audio_npy_path, text_txt_path = None, None, None
        process_file(file_path, frame_npy_path, audio_npy_path, text_txt_path, image_jpg_path)

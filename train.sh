# Confirm Position of the code
CONDA_NEW_ENV=taac2021-tagging-pytrochyyds
ENV_ROOT=/home/tione/notebook
CODE_ROOT=${ENV_ROOT}/VideoStructuring
CODE_BASE=${CODE_ROOT}/taac2021_tagging_pytorchyyds
DATA_BASE=${CODE_ROOT}/dataset
# #################### get env directories
# CONDA_ROOT
CONDA_CONFIG_ROOT_PREFIX=$(conda config --show root_prefix)
get_conda_root_prefix() {
  TMP_POS=$(awk -v a="${CONDA_CONFIG_ROOT_PREFIX}" -v b="/" 'BEGIN{print index(a, b)}')
  TMP_POS=$((TMP_POS-1))
  if [ $TMP_POS -ge 0 ]; then
    echo "${CONDA_CONFIG_ROOT_PREFIX:${TMP_POS}}"
  else
    echo ""
  fi
}
CONDA_ROOT=$(get_conda_root_prefix)
if [ ! -d "${CONDA_ROOT}" ]; then
  echo "[$(date "+%Y-%m-%d %H:%M:%S")] ERROR 找不到Conda环境：${CONDA_ROOT}"
  exit 1
fi

if [ ! -d "${DATA_BASE}" ]; then
  echo "[$(date "+%Y-%m-%d %H:%M:%S")] ERROR 找不到数据集：${DATASET_ROOT}"
  exit 1
fi

CONDA_CONFIG_FILE="${CONDA_ROOT}/etc/profile.d/conda.sh"
if [ ! -f "${CONDA_CONFIG_FILE}" ]; then
  echo "[$(date "+%Y-%m-%d %H:%M:%S")] ERROR 找不到Conda配置文件：${CONDA_CONFIG_FILE}"
  exit 1
fi
source "${CONDA_CONFIG_FILE}"
conda activate ${ENV_ROOT}/envs/${CONDA_NEW_ENV}

check_gpu=$(python -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; import tensorflow as tf; print(tf.test.is_gpu_available())")
if [ "${check_gpu}" == "False" ]; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] ERROR Conda环境启动失败！"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 已启动Conda环境！"


echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 开始拷贝ASR和OCR文本特征..."  # 预计用时30秒
cp ${ENV_ROOT}/algo-2021/dataset/tagging/tagging_dataset_train_5k/text_txt/tagging/ ${CODE_ROOT}/dataset/tagging/tagging_dataset_train_5k/text_txt/ -r
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 文本特征已就绪！"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 开始拷贝音频特征..."  # 预计用时1分钟
cp ${ENV_ROOT}/algo-2021/dataset/tagging/tagging_dataset_train_5k/audio_npy/Vggish/tagging/ ${CODE_ROOT}/dataset/tagging/tagging_dataset_train_5k/audio_npy/Vggish/ -r
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 音频特征已就绪！"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 开始抽取Video特征..."   # 预计用时1小时30分钟
rm -rf ${DATA_BASE}/tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging/
mkdir ${DATA_BASE}/tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging/
python ${CODE_BASE}/preprocess/feat_extract_main.py     --test_files_dir ${DATA_BASE}/videos/video_5k/train_5k     --frame_npy_folder ${DATA_BASE}/tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging/            --audio_npy_folder ${DATA_BASE}/tagging/tagging_dataset_train_5k/audio_npy     --image_jpg_folder ${DATA_BASE}/tagging/tagging_dataset_train_5k/image_jpg     --text_txt_folder ${DATA_BASE}/tagging/tagging_dataset_train_5k/text_txt     --datafile_path ${DATA_BASE}/tagging/GroundTruth/datafile/train.txt     --extract_type 1     --image_batch_size 300     --imgfeat_extractor efficientnet | grep -v "I tensorflow" >> /home/tione/notebook/log/feat_extract.log &
sleep 30s
python ${CODE_BASE}/preprocess/feat_extract_main.py     --test_files_dir ${DATA_BASE}/videos/video_5k/train_5k     --frame_npy_folder ${DATA_BASE}/tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging/            --audio_npy_folder ${DATA_BASE}/tagging/tagging_dataset_train_5k/audio_npy     --image_jpg_folder ${DATA_BASE}/tagging/tagging_dataset_train_5k/image_jpg     --text_txt_folder ${DATA_BASE}/tagging/tagging_dataset_train_5k/text_txt     --datafile_path ${DATA_BASE}/tagging/GroundTruth/datafile/train.txt     --extract_type 1     --image_batch_size 300     --imgfeat_extractor efficientnet | grep -v "I tensorflow" >> /home/tione/notebook/log/feat_extract.log &
sleep 30s
python ${CODE_BASE}/preprocess/feat_extract_main.py     --test_files_dir ${DATA_BASE}/videos/video_5k/train_5k     --frame_npy_folder ${DATA_BASE}/tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging/            --audio_npy_folder ${DATA_BASE}/tagging/tagging_dataset_train_5k/audio_npy     --image_jpg_folder ${DATA_BASE}/tagging/tagging_dataset_train_5k/image_jpg     --text_txt_folder ${DATA_BASE}/tagging/tagging_dataset_train_5k/text_txt     --datafile_path ${DATA_BASE}/tagging/GroundTruth/datafile/train.txt     --extract_type 1     --image_batch_size 300     --imgfeat_extractor efficientnet --do_logging 1 | grep -v "I tensorflow" >> /home/tione/notebook/log/feat_extract.log 

echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 开始检查Video特征抽取结果..."
python ${CODE_BASE}/preprocess/feat_extract_main.py     --test_files_dir ${DATA_BASE}/videos/video_5k/train_5k     --frame_npy_folder ${DATA_BASE}/tagging/tagging_dataset_train_5k/video_npy/Youtube8M/tagging/            --audio_npy_folder ${DATA_BASE}/tagging/tagging_dataset_train_5k/audio_npy     --image_jpg_folder ${DATA_BASE}/tagging/tagging_dataset_train_5k/image_jpg     --text_txt_folder ${DATA_BASE}/tagging/tagging_dataset_train_5k/text_txt     --datafile_path ${DATA_BASE}/tagging/GroundTruth/datafile/train.txt     --extract_type 1     --image_batch_size 300     --imgfeat_extractor efficientnet --do_logging 1 | grep -v "I tensorflow" >> ${ENV_ROOT}/log/feat_extract.log  
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 视频特征抽取完成！"


echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 开始准备K折训练数据..."
python ${CODE_BASE}/utils/k_fold_prepare.py ${DATA_BASE}/tagging/GroundTruth/datafile/train.txt ${DATA_BASE}/tagging/GroundTruth/datafile/val.txt ${DATA_BASE}/tagging/GroundTruth/datafile/train_{}.txt ${DATA_BASE}/tagging/GroundTruth/datafile/valid_{}.txt ${CODE_BASE}/configs/
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO K折训练数据已就绪！"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 开始进行K折训练..."
for fold in 0 1 2 3 4 5 6 7 8 9
do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 开始训练第${fold}个模型..."
    python ${CODE_BASE}/train.py --config "${CODE_BASE}/configs/config.tagging.5k.$fold.yaml" > ${ENV_ROOT}/log/train_log_$fold.txt
    BEST_MODEL=$(ls -td -- ${CODE_BASE}/checkpoints/tagging5k_temp/export/* | head -n 1)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 第${fold}个模型训练完成：$BEST_MODEL"
    python ./utils/save_best_ckpt.py ${CODE_BASE}/checkpoints/tagging5k_temp/
    rm -rf ${CODE_ROOT}/KFoldModels/model_"$fold" 
    mkdir ${CODE_ROOT}/KFoldModels/model_"$fold"
    cp -r ${CODE_BASE}/checkpoints/tagging5k_temp/* ${CODE_ROOT}/KFoldModels/model_"$fold"
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO K折训练已完成！"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 模型训练已完成！请运行sudo chmod a+x ./infer.sh && ./infer.sh进行最终的模型预测！"


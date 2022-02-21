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

check_gpu=$(python -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; import tensorflow.compat.v1 as tf; print(tf.test.is_gpu_available())")
if [ "${check_gpu}" == "False" ]; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] ERROR Conda环境启动失败！"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 已启动Conda环境！"


echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 开始拷贝ASR和OCR文本特征..."
cp ${ENV_ROOT}/algo-2021/dataset/tagging/tagging_dataset_test_5k_2nd/text_txt/tagging/ ${CODE_ROOT}/dataset/tagging/tagging_dataset_test_5k_2nd/text_txt/ -r
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 文本特征已就绪！"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 开始拷贝音频特征..."
cp ${ENV_ROOT}/algo-2021/dataset/tagging/tagging_dataset_test_5k_2nd/audio_npy/Vggish/tagging/ ${CODE_ROOT}/dataset/tagging/tagging_dataset_test_5k_2nd/audio_npy/Vggish/ -r
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 音频特征已就绪！"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 开始抽取Video特征..."
python ${CODE_BASE}/preprocess/feat_extract_main.py     --test_files_dir ${DATA_BASE}/videos/test_5k_2nd/     --frame_npy_folder ${DATA_BASE}/tagging/tagging_dataset_test_5k_2nd/video_npy/Youtube8M/tagging/            --audio_npy_folder ${DATA_BASE}/tagging/tagging_dataset_test_5k_2nd/audio_npy     --image_jpg_folder ${DATA_BASE}/tagging/tagging_dataset_test_5k_2nd/image_jpg     --text_txt_folder ${DATA_BASE}/tagging/tagging_dataset_test_5k_2nd/text_txt     --datafile_path ${DATA_BASE}/tagging/GroundTruth/datafile/train.txt     --extract_type 1     --image_batch_size 300     --imgfeat_extractor efficientnet | grep -v "I tensorflow" >> /home/tione/notebook/log/feat_extract.log &
sleep 30s
python ${CODE_BASE}/preprocess/feat_extract_main.py     --test_files_dir ${DATA_BASE}/videos/test_5k_2nd/     --frame_npy_folder ${DATA_BASE}/tagging/tagging_dataset_test_5k_2nd/video_npy/Youtube8M/tagging/            --audio_npy_folder ${DATA_BASE}/tagging/tagging_dataset_test_5k_2nd/audio_npy     --image_jpg_folder ${DATA_BASE}/tagging/tagging_dataset_test_5k_2nd/image_jpg     --text_txt_folder ${DATA_BASE}/tagging/tagging_dataset_test_5k_2nd/text_txt     --datafile_path ${DATA_BASE}/tagging/GroundTruth/datafile/train.txt     --extract_type 1     --image_batch_size 300     --imgfeat_extractor efficientnet | grep -v "I tensorflow" >> /home/tione/notebook/log/feat_extract.log &
sleep 30s
python ${CODE_BASE}/preprocess/feat_extract_main.py     --test_files_dir ${DATA_BASE}/videos/test_5k_2nd/     --frame_npy_folder ${DATA_BASE}/tagging/tagging_dataset_test_5k_2nd/video_npy/Youtube8M/tagging/            --audio_npy_folder ${DATA_BASE}/tagging/tagging_dataset_test_5k_2nd/audio_npy     --image_jpg_folder ${DATA_BASE}/tagging/tagging_dataset_test_5k_2nd/image_jpg     --text_txt_folder ${DATA_BASE}/tagging/tagging_dataset_test_5k_2nd/text_txt     --datafile_path ${DATA_BASE}/tagging/GroundTruth/datafile/train.txt     --extract_type 1     --image_batch_size 300     --imgfeat_extractor efficientnet --do_logging 1 | grep -v "I tensorflow" >> /home/tione/notebook/log/feat_extract.log 

echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 开始检查Video特征抽取结果..."
python ${CODE_BASE}/preprocess/feat_extract_main.py     --test_files_dir ${DATA_BASE}/videos/test_5k_2nd/     --frame_npy_folder ${DATA_BASE}/tagging/tagging_dataset_test_5k_2nd/video_npy/Youtube8M/tagging/            --audio_npy_folder ${DATA_BASE}/tagging/tagging_dataset_test_5k_2nd/audio_npy     --image_jpg_folder ${DATA_BASE}/tagging/tagging_dataset_test_5k_2nd/image_jpg     --text_txt_folder ${DATA_BASE}/tagging/tagging_dataset_test_5k_2nd/text_txt     --datafile_path ${DATA_BASE}/tagging/GroundTruth/datafile/train.txt     --extract_type 1     --image_batch_size 300     --imgfeat_extractor efficientnet --do_logging 1 | grep -v "I tensorflow" >> /home/tione/notebook/log/feat_extract.log 

echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 视频特征抽取完成！"


echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 开始进行预测..."
for fold in 0 2 4 6 8 
do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 第${fold}个子模型开始预测..."
    rm -rf ${CODE_ROOT}/KFoldResults/test/results_"$fold"/
    mkdir ${CODE_ROOT}/KFoldResults/test/results_"$fold"/
    chmod 777 ${CODE_ROOT}/KFoldResults/test/results_"$fold"/
    BEST_MODEL=$(ls -td -- ${CODE_ROOT}/KFoldModels/model_"$fold"/export/* | head -n 1)
    python ${CODE_BASE}/infer.py  \
                    --model_pb ${BEST_MODEL} \
                    --tag_id_file ${CODE_ROOT}/dataset/label_id.txt \
                    --test_dir ${CODE_ROOT}/dataset/videos/test_5k_2nd/ \
                    --output_json ${CODE_ROOT}/KFoldResults/test/results_"$fold"/tagging_5k.json \
                    --load_feat 1 \
                    --feat_dir ${CODE_ROOT}/dataset/tagging/tagging_dataset_test_5k_2nd/  \
                    --top_k 82 > ${ENV_ROOT}/log/test_eval_log.txt &
                    
    sleep 10s
    fold=$(($fold+1))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 第${fold}个子模型开始预测..."
    rm -rf ${CODE_ROOT}/KFoldResults/test/results_"$fold"/
    mkdir ${CODE_ROOT}/KFoldResults/test/results_"$fold"/
    chmod 777 ${CODE_ROOT}/KFoldResults/test/results_"$fold"/
    BEST_MODEL=$(ls -td -- ${CODE_ROOT}/KFoldModels/model_"$fold"/export/* | head -n 1)
    python ${CODE_BASE}/infer.py  \
                    --model_pb ${BEST_MODEL} \
                    --tag_id_file ${CODE_ROOT}/dataset/label_id.txt \
                    --test_dir ${CODE_ROOT}/dataset/videos/test_5k_2nd/ \
                    --output_json ${CODE_ROOT}/KFoldResults/test/results_"$fold"/tagging_5k.json \
                    --load_feat 1 \
                    --feat_dir ${CODE_ROOT}/dataset/tagging/tagging_dataset_test_5k_2nd/  \
                    --top_k 82 > ${ENV_ROOT}/log/test_eval_log.txt
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO K折预测已完成！"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 开始进行模型融合..."
python ${CODE_BASE}/utils/k_fold_fusion.py 10 ${CODE_ROOT}/dataset/label_id.txt ${CODE_ROOT}/KFoldResults/test/results_{}/tagging_5k.json ${CODE_BASE}/results/tagging_5k.json 20
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 模型融合结果已完成！"

cp ${CODE_BASE}/results/tagging_5k.json ${ENV_ROOT}/pytorchyyds_prediction_5k.json
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO 'Pytorch永远滴神'团队最终预测结果已保存到：${ENV_ROOT}/pytorchyyds_prediction_5k.json"

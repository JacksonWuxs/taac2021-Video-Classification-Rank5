# Confirm Position of the code
CONDA_NEW_ENV=taac2021-tagging-pytrochyyds
ENV_ROOT=/home/tione/notebook
CODE_ROOT=${ENV_ROOT}/VideoStructuring
CODE_BASE=${CODE_ROOT}/taac2021_tagging_pytorchyyds

if [ ! -d "${ENV_ROOT}" ]; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] ERROR 不存在环境根目录：${ENV_ROOT}"
    exit 1
fi
if [ ! -d "${CODE_ROOT}" ]; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] ERROR 不存在项目根目录：${CODE_ROOT}"
    exit 1
fi
if [ ! -d "${CODE_BASE}" ]; then
  echo "[$(date "+%Y-%m-%d %H:%M:%S")] ERROR 当前代码未拷贝到目录：${CODE_BASE}"
  exit 1
fi
if [ ! "$(pwd)" = "${CODE_BASE}" ]; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] ERROR 请将代码文件夹复制到${CODE_BASE}，并确保工作路径为：${CODE_BASE}"
    exit 1
fi

# Copy Data from Original Shared Folder
echo "[$(date "+%Y-%m-%d %H:%M:%S")] INFO 开始拷贝数据到本地..." # 预计17分钟
rm -rf /home/tione/notebook/VideoStructuring/dataset
mkdir /home/tione/notebook/VideoStructuring/dataset
cp /home/tione/notebook/algo-2021/dataset/label_id.txt /home/tione/notebook/VideoStructuring/dataset/

mkdir /home/tione/notebook/VideoStructuring/dataset/videos
cp /home/tione/notebook/algo-2021/dataset/videos/video_5k /home/tione/notebook/VideoStructuring/dataset/videos -r
cp /home/tione/notebook/algo-2021/dataset/videos/test_5k_2nd /home/tione/notebook/VideoStructuring/dataset/videos -r

mkdir /home/tione/notebook/VideoStructuring/dataset/pretrained_models
cp /home/tione/notebook/algo-2021/dataset/pretrained_models/* /home/tione/notebook/VideoStructuring/dataset/pretrained_models -r

mkdir /home/tione/notebook/VideoStructuring/dataset/tagging
mkdir /home/tione/notebook/VideoStructuring/dataset/tagging/GroundTruth
cp /home/tione/notebook/algo-2021/dataset/tagging/GroundTruth/* /home/tione/notebook/VideoStructuring/dataset/tagging/GroundTruth -r
mkdir /home/tione/notebook/VideoStructuring/dataset/tagging/tagging_dataset_train_5k/
mkdir /home/tione/notebook/VideoStructuring/dataset/tagging/tagging_dataset_train_5k/audio_npy/
mkdir /home/tione/notebook/VideoStructuring/dataset/tagging/tagging_dataset_train_5k/audio_npy/Vggish
mkdir /home/tione/notebook/VideoStructuring/dataset/tagging/tagging_dataset_train_5k/text_txt
mkdir /home/tione/notebook/VideoStructuring/dataset/tagging/tagging_dataset_train_5k/video_npy/
mkdir /home/tione/notebook/VideoStructuring/dataset/tagging/tagging_dataset_train_5k/video_npy/Youtube8M
mkdir /home/tione/notebook/VideoStructuring/dataset/tagging/tagging_dataset_test_5k_2nd/
mkdir /home/tione/notebook/VideoStructuring/dataset/tagging/tagging_dataset_test_5k_2nd/audio_npy/
mkdir /home/tione/notebook/VideoStructuring/dataset/tagging/tagging_dataset_test_5k_2nd/audio_npy/Vggish
mkdir /home/tione/notebook/VideoStructuring/dataset/tagging/tagging_dataset_test_5k_2nd/text_txt
mkdir /home/tione/notebook/VideoStructuring/dataset/tagging/tagging_dataset_test_5k_2nd/video_npy
mkdir /home/tione/notebook/VideoStructuring/dataset/tagging/tagging_dataset_test_5k_2nd/video_npy/Youtube8M
echo "[$(date "+%Y-%m-%d %H:%M:%S")] INFO 数据拷贝完成！"

echo "[$(date "+%Y-%m-%d %H:%M:%S")] INFO 开始配置系统环境..." # 预计1.5分钟
CONDA_CONFIG_ROOT_PREFIX=$(conda config --show root_prefix)
get_conda_root_prefix() {
    TMP_POS=$(awk -v a="${CONDA_CONFIG_ROOT_PREFIX}" -v b="/" 'BEGIN{print index(a, b)}')
    TMP_POS=$((TMP_POS-1))
    if [ $TMP_POS -ge 0 ]; then
      echo "${CONDA_CONFIG_ROOT_PREFIX:${TMP_POS}}"
    fi
}
CONDA_ROOT=$(get_conda_root_prefix)
if [ ! -d "${CONDA_ROOT}" ]; then
  echo "[$(date "+%Y-%m-%d %H:%M:%S")] ERROR 未检测到CONDA根目录：${CONDA_ROOT}"
  exit 1
fi

OS_ID=$(awk -F= '$1=="ID" { print $2 ;}' /etc/os-release)
OS_ID=${OS_ID//"\""/""}

if [ "${OS_ID}" == "ubuntu" ]; then
    sudo apt-get update
    sudo apt-get install -y apt-utils libsndfile1-dev ffmpeg
elif [ "${OS_ID}" == "centos" ]; then
    yum install -y libsndfile libsndfile-devel ffmpeg ffmpeg-devel
else
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] ERROR 不支持的操作系统：${OS_ID}"
    exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

conda create --prefix ${ENV_ROOT}/envs/${CONDA_NEW_ENV} -y cudatoolkit=10.0 cudnn=7.6.0 python=3.7 ipykernel
conda activate ${ENV_ROOT}/envs/${CONDA_NEW_ENV}

python -m ipykernel install --user --name ${CONDA_NEW_ENV} --display-name "TAAC2021 (${CONDA_NEW_ENV})"
echo "[$(date "+%Y-%m-%d %H:%M:%S")] INFO 系统环境配置完成！"


echo "[$(date "+%Y-%m-%d %H:%M:%S")] INFO 开始下载第三方Python环境..."
pip config set global.index-url https://mirrors.tencent.com/pypi/simple/
pip install -r requirement.txt
pip install tensorflow-gpu==1.14 efficientnet opencv-python torch==1.2.0 scikit-learn jieba
check_env=$(python -c """
try:
    import tensorflow as tf, cv2, torch, efficientnet, sklearn
    print('[TensorFlow]', tf.__version__, '[Torch]', torch.__version__, '[EfficientNet]', efficientnet.__version__, '[OpenCV]', cv2.__version__, '[ScikitLearn]', sklearn.__version__)
except Exception as e:
    print('环境安装存在异常！请重新执行init脚本！')
""")
if [ "${check_env}" == "环境安装存在异常！请重新执行init脚本！" ]; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] ERROR ${check_env}"
    exit 1
fi
sed -i "s/\.decode('utf8')//g" /home/tione/notebook/envs/taac2021-tagging-pytrochyyds/lib/python3.7/site-packages/tensorflow/python/keras/saving/hdf5_format.py
echo "[$(date "+%Y-%m-%d %H:%M:%S")] INFO 第三方Python环境已安装完毕！"

echo "[$(date "+%Y-%m-%d %H:%M:%S")] INFO 开始下载预训练模型..."
rm -rf ${CODE_BASE}/pretrained
wget https://algo-tencent-2021-1256646044.cos.ap-guangzhou.myqcloud.com/pretrained_models/pretrained.zip
unzip pretrained.zip
rm -rf pretrained.zip
echo "[$(date "+%Y-%m-%d %H:%M:%S")] INFO 预训练模型下载完成！"

rm -rf ${ENV_ROOT}/log
mkdir ${ENV_ROOT}/log
rm -rf ${CODE_BASE}/results
mkdir ${CODE_BASE}/results
rm -rf ${CODE_BASE}/checkpoints
mkdir ${CODE_BASE}/checkpoints
rm -rf ${CODE_ROOT}/KFoldModels
mkdir ${CODE_ROOT}/KFoldModels
rm -rf ${CODE_ROOT}/KFoldResults
mkdir ${CODE_ROOT}/KFoldResults
mkdir ${CODE_ROOT}/KFoldResults/train
mkdir ${CODE_ROOT}/KFoldResults/test
echo "[$(date "+%Y-%m-%d %H:%M:%S")] INFO 系统初始化完成！请运行sudo chmod a+x ./train.sh && ./train.sh进行K折模型训练！"
## 2021腾讯广告算法大赛-赛道二-第五名解决方案

> 作者：吴烜圣 ([email](wuxsmail@163.com), [phone](13036864606))、杨非池 ([email](feichi.yang@usc.edu)）、周童（[email](zhoutong0322@163.com)）、林心悦（[email](xl9yr@virginia.edu)）

#### 0 代码复现

* 推荐配置：CPU: 6 Cores  Memory: 16GB  GPU:V100-32GB

* 配置环境：

  在完成下述“步骤0”至“步骤4”之后，将得到如下目录：

  ```shell
  /home/tione/notebook/
  ├── algo-2021
  ├── envs                            
  │   └── taac2021-tagging-pytrochyyds # 本项目的conda环境（init.sh自动创建）
  ├── log                              # 运行日志（init.sh自动创建）
  └── VideoStructuring                 # 项目路径（请手动创建）
      ├── dataset                      # 数据和特征（init.sh自动创建）
      ├── KFoldModels                  # K折模型参数（init.sh自动创建）
      ├── KFoldResults                 # 单折预测结果（init.sh自动创建）
      └── taac2021_tagging_pytorchyyds # 项目的代码（请手动创建并拷贝代码到此处）
          ├── init.sh                  # 环境初始化脚本
          ├── train.sh                 # 训练模型脚本
          ├── infer.sh                 # 模型推断脚本
          ├── pretrained               # 预训练模型权重（init.sh自动下载）
          ├── checkpoints              # K折模型训练ckpt（init.sh自动创建）
          ├── results                  # k这模型预测（init.sh自动创建）
          ├── configs
          ├── infer.py
          ├── preprocess
          ├── readme.md
          ├── requirement.txt     
          ├── src
          ├── train.py
          └── utils
  ```

  * **步骤0：** 空的机器仅包含`/home/tione/notebook/algo-2021`一个文件夹

  * **步骤1：** 创建本项目的文件夹：`mkdir /home/tione/notebook/VideoStructuring`，除了下一步骤创建的代码目录外，`init.sh`脚本还会在此目录下自动创建存放数据的`dataset`文件夹、存放模型的`KFoldModels`文件夹和存放K这交叉结果的`KFoldResults`文件夹。

  * **步骤2：** 创建本项目的代码目录：`mkdir home/tione/notebook/VideoStructuring/taac2021_tagging_pytorchyyds`，并 **将项目的所有代码移动到此路径下** ，确保`init.sh`、`train.sh`和`infer.sh`三个文件位于该文件夹中。

  * **步骤3：** `cd /home/tione/notebook/VideoStructuring/taac2021_tagging_pytorchyyds`

  * **步骤4：** `sudo chmod a+x ./init.sh && ./init.sh`

    ```shell
    shell> sudo chmod a+x ./init.sh && ./init.sh
    [2021-07-05 18:50:24] INFO 开始拷贝数据到本地...
    [2021-07-05 19:07:20] INFO 数据拷贝完成！
    [2021-07-05 19:07:20] INFO 开始配置系统环境...
    [2021-07-05 19:09:23] INFO 系统环境配置完成！
    [2021-07-05 19:09:23] INFO 开始下载第三方Python环境...
    [2021-07-05 19:12:21] INFO 第三方Python环境已安装完毕！
    [2021-07-05 19:12:21] INFO 开始下载预训练模型...
    [2021-07-05 19:16:16] INFO 预训练模型下载完成！
    [2021-07-05 19:16:16] INFO 系统初始化完成！请运行sudo chmod a+x ./train.sh && ./train.sh进行K折模型训练！
    ```

* 训练集的特征抽取和K折模型训练：
  * **步骤5：** `sudo chmod a+x ./train.sh && ./train.sh`
  
    ```shell
    shell> sudo chmod a+x ./train.sh && ./train.sh
    [2021-07-05 19:25:52] INFO 已启动Conda环境！
    [2021-07-05 19:25:52] INFO 开始拷贝ASR和OCR文本特征...
    [2021-07-05 19:26:16] INFO 文本特征已就绪！
    [2021-07-05 19:26:16] INFO 开始拷贝音频特征...
    [2021-07-05 19:26:39] INFO 音频特征已就绪！
    [2021-07-05 19:26:39] INFO 开始抽取Video特征...
    [2021-07-05 20:54:48] INFO 开始检查Video特征抽取结果...
    [2021-07-05 20:55:09] INFO 视频特征抽取完成！
    [2021-07-05 20:55:09] INFO 开始准备K折训练数据...
    [2021-07-05 20:55:09] INFO K折训练数据已就绪！
    [2021-07-05 20:55:09] INFO 开始进行K折训练...
    [2021-07-05 20:55:09] INFO 开始训练第0个模型...
    # ... 此处省略10个模型的训练日志 ....
    [2021-07-06 07:50:06] INFO 第9个模型训练完成！
    [2021-07-06 07:50:06] INFO K折训练已完成！
    [2021-07-06 07:50:06] INFO 模型训练已完成！请运行sudo chmod a+x ./infer.sh && ./infer.sh进行最终的模型预测！
    ```
  
* 测试集的特征抽取和K折模型预测：
  * **步骤6：** `sudo chmod a+x ./infer.sh && ./infer.sh`
  
    ```shell
    shell> sudo chmod a+x ./infer.sh && ./infer.sh
    [2021-07-06 14:49:02] INFO 已启动Conda环境！
    [2021-07-06 14:49:23] INFO 开始拷贝ASR和OCR文本特征...
    [2021-07-06 14:49:51] INFO 文本特征已就绪！
    [2021-07-06 14:49:51] INFO 开始拷贝音频特征...
    [2021-07-06 14:50:29] INFO 音频特征已就绪！
    [2021-07-06 14:50:29] INFO 开始抽取Video特征...
    [2021-07-06 15:16:48] INFO 视频特征抽取完成！
    [2021-07-06 15:16:48] INFO 开始进行预测...
    [2021-07-06 15:16:48] INFO 第0个子模型开始预测...
    # ... 此处省略10个模型的预测日志 ...
    [2021-07-06 15:37:00] INFO K折预测已完成！
    [2021-07-06 15:37:00] INFO 开始进行模型融合...
    [2021-07-06 15:37:05] INFO 模型融合结果已完成！
    [2021-07-06 15:37:05] INFO 'Pytorch永远滴神'团队最终预测结果已保存到：/home/tione/notebook/pytorchyyds_prediction_5k.json
    ```
  
  * **预测结果：** `/home/tione/notebook/pytorchyyds_prediction_5k.json`

#### 1 预训练模型

​	   模型仅使用了Video、Text、Audio三种模态，没有使用视频中间帧的Image模态。除此以外，Text流的文本表征除了Chinese Bert以外，我们还使用腾讯AI实验室的预训练词向量，用于增强Video模态中每一帧的表征。**所有需要的预训练模型已经传到[COS对象存储](https://algo-tencent-2021-1256646044.cos.ap-guangzhou.myqcloud.com/pretrained_models/pretrained.zip)，在init.sh过程中会自行下载并解压。**

* 与Baseline相同的预训练模型：
  1. Audio模态：Vggish
  2. Text模态：ChineseBert-base

* 与Baseline不同的预训练模型
  1. Video模态：EfficientNet-B5-NoisyStudent（[Code](https://github.com/qubvel/efficientnet)、[Paper](https://arxiv.org/pdf/1905.11946.pdf)）
  2. Image模态：该模态被丢弃，未使用任何预训练模型
  3. Text模态：腾讯AI Lab预训练词向量 （[Code](https://ai.tencent.com/ailab/nlp/zh/embedding.html)、[Paper]([Embedding Dataset -- NLP Center, Tencent AI Lab](https://ai.tencent.com/ailab/nlp/zh/embedding.html))）

#### 2 预计用时

* 初始化阶段：`sudo chmod a+x ./init.sh && ./init.sh` （大约30分钟）
  1. 复制原始视频数据到本地：20分钟
  2. 安装系统环境：3分钟
  3. 安装Python环境：3分钟
  4. 安装预训练模型：5分钟
* 训练阶段：`sudo chmod a+x ./train.sh && ./train.sh` （11小时 + 5k训练集文本抽取标准时间 + 5k训练集Vggish特征抽取标准时间）
  1. 复制训练集Text到本地：baseline标准时间
  2. 复制训练集Audio特征到本地：baseline标准时间
  3. 抽取训练集Video特征：约1小时30分钟
  4. 训练K折模型：约9小时30分钟
* 测试阶段：`sudo chmod a+x ./infer.sh && ./infer.sh` (1小时50分钟 + 5k测试集文本抽取标准时间 + 5k测试集Vggish特征抽取标准时间)

  1. 复制测试集Text到本地：baseline标准时间
  2. 复制测试集Audio特征到本地：baseline标准时间
  3. 抽取测试集Video特征：约1小时25分钟
  4. K折模型预测：约20分钟
  5. K折模型结果融合：5秒


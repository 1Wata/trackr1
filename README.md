# trackr1

## Step1. 数据集下载和环境配置：

我已经将 lasot，TNL2L，OTB_lang 已经全部上传到 modelscope 上了，下载可以如下下载：

```python
#验证SDK token
from modelscope.hub.api import HubApi
api = HubApi()
api.login('470dc5bd-b4d9-4ed2-997b-a3684903260e')

#数据集下载
from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('FineTuneWata/tracking_data')
#您可按需配置 subset_name、split，参照“快速使用”示例代码
```

下载之后目录结构是这样的，需要进入每一个数据集内部将当前路径下的 zip 文件全部解压，最终路径示例如下：

```shell
 -- data
     -- lasot
         |-- airplane
         |-- basketball
         |-- bear
         ...
     -- TNL2K
         |-- TNL2K_test_subset
         |-- TNL2K_train_subset
         ...
     -- otb_lang
         |-- OTB_query_test
         |-- OTB_query_train
         |-- OTB_videos

```



环境配置：

代码仓库已经上传到 [1Wata/mllmTracker](https://github.com/1Wata/mllmTracker) 中了，环境配置如下：

```bash
git clone https://github.com/1Wata/mllmTracker.git
cd mllmTracker

conda create -n track python=3.10 -y
conda activate track
conda install cudatoolkit=12.4 -c nvidia
pip install -r requirements.txt
```

克隆代码仓库之后还需要配置一下数据集的路径，就在 `dataset_interface/dataset_config.yaml` 路径下，修改三个数据集路径就行（因为目前只用了三个数据集）；除此之外还要把 workspace_dir 改为整个仓库的路径（mllmTracker 代码仓库的路径）

```yaml
env:
  workspace_dir: '/data1/lihaobo/tracking/'
  lasot_dir: '/data1/lihaobo/tracking/data/lasot'
  tnl2k_dir: '/data1/lihaobo/tracking/data/TNL2K_CVPR2021'
  otb_lang_dir: '/data1/lihaobo/tracking/data/OTB_lang'
```

## Step2. 数据集制作：

```shell
cd dataset_interface
python make_rft_dataset.py --samples_per_epoch 40000 --output_dir path/to/save/json_dataset
```
再运行 convert_parquet.py 文件制作parquet数据集
## Step3. 训练

开始训练，首先要下载 Qwen2.5VL-3B 模型，使用 EasyR1/examples/config.yaml 脚本即可训练

 

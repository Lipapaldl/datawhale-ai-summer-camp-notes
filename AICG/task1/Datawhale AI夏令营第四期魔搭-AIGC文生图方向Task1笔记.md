# Datawhale AI夏令营第四期魔搭-AIGC文生图方向Task1笔记

> 打卡目标：
>
> 1. 学习文生图相关原理
> 2. 熟悉魔搭相关文生图工具&应用
> 3. 完成可图故事赛体验

[TOC]

___

# 一、文生图历史发展——知其然知其所以然

## 1.1 生成模型发展历程

![image-20240808203505738](IMG\image-20240808203505738.png)

_____

###  生成模型的核心出装：**拟合数据分布**

如何获得生成模型？

- step0：建立先验空间Z（eg.高斯分布）

- step1：建立传输函数（or 积分核）的神经网络参数化模型 

- $$
  f(z) = f_θ(z)
  $$

- step2：建立分布间距离的度量

- $$
  d(p_1,p_2)
  $$

- step3：寻找优化算法求解最优化问题

- 

- $$
  min_θ{d(p_{f_θ}(z),p_{data})}
  $$

- 

___

###  生成模型对比图

![image-20240808204447624](IMG\image-20240808204447624.png)

| 生成模型                      | 分布间度量           | 优化算法          |
| ----------------------------- | -------------------- | ----------------- |
| GAN 生成对抗网络              | Jansen-Shannon散度   | Min-Max优化 + SGD |
| VAE 变分自编码                | KL散度               | 变分法 + SGD      |
| Normalizing Flow 标准化流模型 | 重构损失（对数似然） | SGD               |
| Diffusion Model 去噪扩散模型  | KL散度               | 变分法 + SGD      |

http://t.csdnimg.cn/i2xQi 【通俗理解生成对抗网络GAN】

http://t.csdnimg.cn/3YEyR  【VAE 模型基本原理简单介绍】

http://t.csdnimg.cn/cSFrv 【通俗形象地分析比较生成模型（GAN/VAE/Flow/Diffusion/AR）】

> 通过一个比喻来说明它们之间的区别。我们把数据的生成过程，也就是从Z映射到X的过程，比喻为过河。河的左岸是Z，右岸是X，过河就是乘船从左岸码头到达右岸码头。船可以理解为生成模型，码头的位置可以理解为样本点Z或者X在分布空间的位置。不同的生成模型有不同的过河的方法，如下图所示，我们分别来分析。
>
> ![a8cd0b39786f544ba8873a71bad92a3d.jpeg](https://i-blog.csdnimg.cn/blog_migrate/13a3811eeedc07259467dfb0fef35b29.jpeg)
>
> 1. **GAN的过河方式**
>
> 从先验分布随机采样一个Z，也就是在左岸随便找一个码头，直接通过对抗损失的方式强制引导船开到右岸，要求右岸下船的码头和真实数据点在分布层面上比较接近。
>
> 2. **VAE的过河方式**
>
> 1）VAE在过河的时候，不是强制把河左岸的一个随机点拉到河右岸，而是考虑右岸的数据到达河左岸会落在什么样的码头。如果知道右岸数据到达左岸大概落在哪些码头，我们直接从这些码头出发就可以顺利回到右岸了。
>
> 2）由于VAE编码器的输出是一个高斯分布的均值和方差，一个右岸的样本数据X到达河左岸的码头位置不是一个固定点，而是一个高斯分布，这个高斯分布在训练时会和一个先验分布（一般是标准高斯分布）接近。
>
> 3）在数据生成时，从先验分布采样出来的Z也大概符合右岸过来的这几个码头位置，通过VAE解码器回到河右岸时，大概能到达真实数据分布所在的码头。
>
> 3. **Flow的过河方式**
>
> 1）Flow的过河方式和VAE有点类似，也是先看看河右岸数据到河左岸能落在哪些码头，在生成数据的时候从这些码头出发，就比较容易能到达河右岸。
>
> 2）和VAE不同的是，对于一个从河右岸码头出发的数据，通过Flow到达河左岸的码头是一个固定的位置，并不是一个分布。而且往返的船开着双程航线，来的时候从什么右岸码头到达左岸码头经过什么路线，回去的时候就从这个左岸码头经过这个路线到达这个右岸码头，是完全可逆的。
>
> 3）Flow需要约束数据到达河左岸码头的位置服从一个先验分布（一般是标准高斯分布），这样在数据生成的时候方便从先验分布里采样码头的位置，能比较好的到达河右岸。
>
> 4. **Diffusion的过河方式**
>
> 1）Diffusion也借鉴了类似VAE和Flow的过河思想，要想到达河右岸，先看看数据从河右岸去到左岸会在哪个码头下船，然后就从这个码头上船，准能到达河右岸的码头。
>
> 2）但是和Flow以及VAE不同的是，Diffusion不只看从右岸过来的时候在哪个码头下船，还看在河中央经过了哪些桥墩或者浮标点。这样从河左岸到河右岸的时候，也要一步一步打卡之前来时经过的这些浮标点，能更好约束往返的航线，确保到达河右岸的码头位置符合真实数据分布。
>
> 3）Diffusion从河右岸过来的航线不是可学习的，而是人工设计的，能保证到达河左岸的码头位置，虽然有些随机性，但是符合一个先验分布（一般是高斯分布），这样方便我们在生成数据的时候选择左岸出发的码头位置。
>
> 4）因为训练模型的时候要求我们一步步打卡来时经过的浮标，在生成数据的时候，基本上也能遵守这些潜在的浮标位置，一步步打卡到达右岸码头。
>
> 5）如果觉得开到河右岸一步步这样打卡浮标有点繁琐，影响船的行进速度，可以选择一次打卡跨好几个浮标，就能加速船行速度，这就对应diffusion的加速采样过程。

## 1.2 扩散模型

### DDPM 开篇之作

Denoising Diffusion Probabilistic Model https://arxiv.org/abs/2006.11239

于2020年发布开创性论文DDPM，展示了扩散模型的能力，图像合成方面击败了GAN，DALL-E2和Imagen 都是基于扩散模型。

### 从概率分布角度理解扩散

![image-20240808212749200](IMG\image-20240808212749200.png)

上图漩涡形状的二维联合概率分布P（x，y），扩散过程q直观理解为 集中有序的点受到噪声扰动向外部扩散，最终变成无序的噪声分布；diffusion model要做的是上述过程的逆过程，将噪声分布N（0，1）逐步去噪以映射到P_data，直到从噪声分布中采样得到想要的图像，完成生成过程。



**后验概率**：贝叶斯统计中，随机事件或不确定事件的后验概率是在考虑和给出相关证据或数据后所得到的条件概率

**马尔可夫链**：状态空间中从一个状态到另一个状态的转换的随机过程，并且下一状态的概率分布只由当前状态决定，在时间序列中和前面的事件均无关

![image-20240808212817163](IMG\image-20240808212817163.png)

**正向扩散过程**：从输入X_0到X_T是一个马尔可夫链，表示状态空间中经过一个到另一个状态的转换的随机过程，上图表示Diffusion Models对应的图像扩散过程。X_T是纯高斯噪声的图片

![image-20240808212131903](IMG\image-20240808212131903.png)

模型训练集中在**逆扩散过程**中，模型训练的目标是学习正向的反过程，即训练概率分布

![image-20240808213202842](IMG\image-20240808213202842.png)

通过沿着马尔可夫链向后遍历，可以重新生成新的数据X_0，完成生成过程。

http://t.csdnimg.cn/qbdyp 【扩散模型 (Diffusion Model) 之最全详解图解】

## 1.3 文生图模型

### Stable Diffusion Models

Stable Diffusion是一个基于Latent Diffusion Models（潜在扩散模型，LDMs）的文图生成（text-to-image）模型。

High-Resolution Image Synthesis with Latent Diffusion Models（2020）

https://arxiv.org/abs/2112.10752

![image-20240808214505982](IMG\image-20240808214505982.png)

现有的扩散模型（Diffusion Model）：

- 通过使用一系列的去噪自编码器，实现了很好的图像合成效果
- 一般的扩散模型需要在像素空间训练和运行，十分消耗计算资源
- 生成一张高分辨率图片，意味着训练空间更高维数，巨大的参数量和高昂训练成本

潜在扩散模型（Latent Diffusion Model）：

- 将图像从像素空间转换到更低维的潜在空间**Latent（通过平衡降低复杂度和保持图像细节，能在保真的同时实现模型加速）**
- 在潜在空间进行相关计算所需的计算量更小
- 最后使用解码器从潜在空间复原到像素空间

___

### checkpoint 主模型

checkpoint 就像游戏关卡存档功能，可以加载保存的模型权重重新开启训练，甚至继续推理

![image-20240808215718060](IMG\image-20240808215718060.png)

___

### Embeding 文本转换

词嵌入，顾名思义就是将文本转换成计算机数字向量，Stable Diffusion中可以使用内置的嵌入模型或创建自定义的嵌入模型来生成嵌入

____

### LoRA 劳拉

Low-Rank Adaptation of Large Language Models，一种微调大模型语言技术。在微调Stable Diffusion中可以用于将图像表示与描述它们的提示相关联的 交叉注意力层。

![image-20240808220128391](IMG\image-20240808220128391.png)

____

### HyperNetwork 风格化

一种生成网络的网络，通过它生成其他网络权重，用于生成描述图像的提示的交叉注意力层

![image-20240808220248765](IMG\image-20240808220248765.png)

### AVE 特定

一种生成网络的网络，通过它来生成其他网络权重。用于生成描述图像的提示的交叉注意力层。

___

## 二、魔塔工具应用

竞赛官网：https://modelscope.cn/brand/view/Kolors?branch=0&tree=0

可图训练资源包 https://www.yuque.com/2ai/model/kdtveg9n5stlmhe7

### 启动环境

step1 进入魔塔官网https://modelscope.cn/ ，点击 我的Notebook -> 配置GPU环境 -> 启动

![image-20240808222619924](IMG\image-20240808222619924.png)

step2 上传文件

![image-20240808222945581](IMG\image-20240808222945581.png)

- baseline.ipynb：LoRA 训练脚本
- ComfyUI.ipynb：ComfyUI 脚本
- kolors_example.json：ComfyUI 工作流（不带 LoRA）
- kolors_with_lora_example.json：ComfyUI 工作流（带 LoRA）

### 运行baseline遇到的问题

项目克隆到本地失败

![image-20240809190211879](IMG\image-20240809190211879.png)

原因：pip版本太旧

解决办法：添加代码然后重启内核重新运行

![deca129695729b65e1ca87640f63b58](C:\Users\xiandan\Documents\WeChat Files\wxid_6lyomvyv1yn722\FileStorage\Temp\deca129695729b65e1ca87640f63b58.png)

继续运行后发现仍然无法解决，尝试将库下载到本地然后上传

![image-20240809192214725](IMG\image-20240809192214725.png)

https://github.com/modelscope/data-juicer.git

https://github.com/modelscope/DiffSynth-Studio.git

因为文件太大上传失败，于是打开科学上网重新运行奇迹般的好了，所以是网络代理问题

____

### 运行baseline

step1 安装Data-Juicer 和 DiffSynth-Studio

```python
import os

!pip install --upgrade pip

!pip install simple-aesthetics-predictor

!git clone https://github.com/modelscope/data-juicer.git

!pip install -v -e data-juicer

!pip uninstall pytorch-lightning -y

!pip install peft lightning pandas torchvision

!git clone https://github.com/modelscope/DiffSynth-Studio.git

!pip install -e DiffSynth-Studio
```

Data-Juicer 是一个一站式**多模态**数据处理系统，旨在为大语言模型 (LLM) 提供更高质量、更丰富、更易“消化”的数据。

![image-20240810090321009](IMG\image-20240810090321009.png)

DiffSynth Studio 是一个扩散引擎。 该项目重组了包括文本编码器、UNet、VAE 等在内的架构，支持[Kolors](https://huggingface.co/Kwai-Kolors/Kolors)模型

___

step2  下载数据集

```python
from modelscope.msdatasets import MsDataset

ds = MsDataset.load(
    'AI-ModelScope/lowres_anime', #项目名称
    subset_name='default', #子数据集名称
    split='train', #提取训练集
    cache_dir="/mnt/workspace/data" #缓存存放地址
)
```

____

step3 使用data-juicer处理图片和元数据，保存到data/lora_dataset/train 目录下

```python
import json, os
from data_juicer.utils.mm_utils import SpecialTokens
from tqdm import tqdm


os.makedirs("./data/lora_dataset/train", exist_ok=True)
os.makedirs("./data/data-juicer/input", exist_ok=True)
with open("./data/data-juicer/input/metadata.jsonl", "w") as f:
    for data_id, data in enumerate(tqdm(ds)):
        image = data["image"].convert("RGB")
        image.save(f"/mnt/workspace/data/lora_dataset/train/{data_id}.jpg")
        metadata = {"text": "二次元", "image": [f"/mnt/workspace/data/lora_dataset/train/{data_id}.jpg"]}
        f.write(json.dumps(metadata))
        f.write("\n")
```

____

step4 使用data-juicer 处理数据

处理的结果保存在 ./data/data-juicer/output/result.jsonl

```python
data_juicer_config = """
# global parameters
project_name: 'data-process'
dataset_path: './data/data-juicer/input/metadata.jsonl'  # path to your dataset directory or file
np: 4  # number of subprocess to process your dataset

text_keys: 'text'
image_key: 'image'
image_special_token: '<__dj__image>'

export_path: './data/data-juicer/output/result.jsonl'

# process schedule
# a list of several process operators with their arguments
process:
    - image_shape_filter:
        min_width: 1024
        min_height: 1024
        any_or_all: any
    - image_aspect_ratio_filter:
        min_ratio: 0.5
        max_ratio: 2.0
        any_or_all: any
"""
with open("data/data-juicer/data_juicer_config.yaml", "w") as file:
    file.write(data_juicer_config.strip())

!dj-process --config data/data-juicer/data_juicer_config.yaml
```

根据result.jsonl，保存处理好的图片 ./data/lora_dataset_processed/train/metadata.csv

```python
import pandas as pd
import os, json
from PIL import Image
from tqdm import tqdm


texts, file_names = [], []
os.makedirs("./data/lora_dataset_processed/train", exist_ok=True)
with open("./data/data-juicer/output/result.jsonl", "r") as file:
    for data_id, data in enumerate(tqdm(file.readlines())):
        data = json.loads(data)
        text = data["text"]
        texts.append(text)
        image = Image.open(data["image"][0])
        image_path = f"./data/lora_dataset_processed/train/{data_id}.jpg"
        image.save(image_path)
        file_names.append(f"{data_id}.jpg")
data_frame = pd.DataFrame()
data_frame["file_name"] = file_names
data_frame["text"] = texts
data_frame.to_csv("./data/lora_dataset_processed/train/metadata.csv", index=False, encoding="utf-8-sig")
data_frame
```

____

step5 下载模型并训练

```python
from diffsynth import download_models

download_models(["Kolors", "SDXL-vae-fp16-fix"])
```

查看训练脚本的输入参数

```cmd
!python DiffSynth-Studio/examples/train/kolors/train_kolors_lora.py -h
```

训练配置

--lora_rank可以控制LoRA模型的参数量，--max_epochs为最大训练批次，

```python
import os

cmd = """
python DiffSynth-Studio/examples/train/kolors/train_kolors_lora.py \
  --pretrained_unet_path models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors \
  --pretrained_text_encoder_path models/kolors/Kolors/text_encoder \
  --pretrained_fp16_vae_path models/sdxl-vae-fp16-fix/diffusion_pytorch_model.safetensors \
  --lora_rank 16 \
  --lora_alpha 4.0 \
  --dataset_path data/lora_dataset_processed \
  --output_path ./models \
  --max_epochs 1 \
  --center_crop \
  --use_gradient_checkpointing \
  --precision "16-mixed" \
""".strip()

os.system(cmd)
```

____

step6 加载模型，输入文本生成图片

```python
from diffsynth import ModelManager, SDXLImagePipeline
from peft import LoraConfig, inject_adapter_in_model
import torch


def load_lora(model, lora_rank, lora_alpha, lora_path):
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out"],
    )
    model = inject_adapter_in_model(lora_config, model)
    state_dict = torch.load(lora_path, map_location="cpu  ")
    model.load_state_dict(state_dict, strict=False)
    return model


# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=[
                                 "models/kolors/Kolors/text_encoder",
                                 "models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors",
                                 "models/kolors/Kolors/vae/diffusion_pytorch_model.safetensors"
                             ])
pipe = SDXLImagePipeline.from_model_manager(model_manager)

# Load LoRA
pipe.unet = load_lora(
    pipe.unet,
    lora_rank=16, # This parameter should be consistent with that in your training script.
    lora_alpha=2.0, # lora_alpha can control the weight of LoRA.
    lora_path="models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt"
)from diffsynth import ModelManager, SDXLImagePipeline
from peft import LoraConfig, inject_adapter_in_model
import torch


def load_lora(model, lora_rank, lora_alpha, lora_path):
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out"],
    )
    model = inject_adapter_in_model(lora_config, model)
    state_dict = torch.load(lora_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    return model


# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=[
                                 "models/kolors/Kolors/text_encoder",
                                 "models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors",
                                 "models/kolors/Kolors/vae/diffusion_pytorch_model.safetensors"
                             ])
pipe = SDXLImagePipeline.from_model_manager(model_manager)

# Load LoRA
pipe.unet = load_lora(
    pipe.unet,
    lora_rank=16, # This parameter should be consistent with that in your training script.
    lora_alpha=2.0, # lora_alpha can control the weight of LoRA.
    lora_path="models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt"
)
```

生成二次元图片

```python
torch.manual_seed(0)
image = pipe(
    prompt="二次元，一个蓝色中分小男孩，在家中沙发上坐着，双手托着腮，很无聊，全身，蓝色背带裤",
    negative_prompt="丑陋、变形、嘈杂、模糊、低对比度",
    cfg_scale=4,
    num_inference_steps=50, height=1024, width=1024,
)
image.save("1.jpg")
torch.manual_seed(1)
image = pipe(
    prompt="二次元，日系动漫，演唱会的观众席，人山人海，一个蓝色中分小男孩穿着蓝色背带裤坐在演唱会的观众席，舞台上衣着华丽的歌星们在唱歌",
    negative_prompt="丑陋、变形、嘈杂、模糊、低对比度",
    cfg_scale=4,
    num_inference_steps=50, height=1024, width=1024,
)
image.save("2.jpg")
torch.manual_seed(2)
image = pipe(
    prompt="二次元，一个蓝色中分小男孩穿着蓝色背带裤坐在演唱会的观众席，露出憧憬的神情",
    negative_prompt="丑陋、变形、嘈杂、模糊、低对比度，色情擦边",
    cfg_scale=4,
    num_inference_steps=50, height=1024, width=1024,
)
image.save("3.jpg")
torch.manual_seed(5)
image = pipe(
    prompt="二次元，一个蓝色中分小男孩穿着蓝色背带裤，对着流星许愿，闭着眼睛，十指交叉，侧面",
    negative_prompt="丑陋、变形、嘈杂、模糊、低对比度，扭曲的手指，多余的手指",
    cfg_scale=4,
    num_inference_steps=50, height=1024, width=1024,
)
image.save("4.jpg")
torch.manual_seed(0)
image = pipe(
    prompt="二次元，一个蓝色中分小男孩穿着蓝色背带裤，在练习室练习唱歌",
    negative_prompt="丑陋、变形、嘈杂、模糊、低对比度",
    cfg_scale=4,
    num_inference_steps=50, height=1024, width=1024,
)
image.save("5.jpg")
torch.manual_seed(1)
image = pipe(
    prompt="二次元，一个蓝色中分小男孩穿着蓝色背带裤，在练习室练习唱歌，手持话筒",
    negative_prompt="丑陋、变形、嘈杂、模糊、低对比度",
    cfg_scale=4,
    num_inference_steps=50, height=1024, width=1024,
)
image.save("6.jpg")
torch.manual_seed(7)
image = pipe(
    prompt="二次元，一个蓝色中分男孩，穿着黑色西装，试衣间，心情忐忑",
    negative_prompt="丑陋、变形、嘈杂、模糊、低对比度",
    cfg_scale=4,
    num_inference_steps=50, height=1024, width=1024,
)
image.save("7.jpg")
torch.manual_seed(0)
image = pipe(
    prompt="二次元，蓝色中分男孩，穿着黑色西装，在台上唱歌",
    negative_prompt="丑陋、变形、嘈杂、模糊、低对比度",
    cfg_scale=4,
    num_inference_steps=50, height=1024, width=1024,
)
image.save("8.jpg")
import numpy as np
from PIL import Image


images = [np.array(Image.open(f"{i}.jpg")) for i in range(1, 9)]
image = np.concatenate([
    np.concatenate(images[0:2], axis=1),
    np.concatenate(images[2:4], axis=1),
    np.concatenate(images[4:6], axis=1),
    np.concatenate(images[6:8], axis=1),
], axis=0)
image = Image.fromarray(image).resize((1024, 2048))
image
```

![image-20240810094413125](IMG\image-20240810094413125.png)


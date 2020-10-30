# Window Detection in Facades Using Heatmaps Fusion
Official implementation of our [paper](http://jcst.ict.ac.cn/EN/10.1007/s11390-020-0253-4).

Chuan-Kang Li, Hong-Xin Zhang, Jia-Xin Liu, Yuan-Qing Zhang, Shan-Chen Zou, Yu-Tong Fang. Window Detection in Facades Using Heatmap Fusion[J].Journal of Computer Science and Technology, 2020, 35(4): 900-912.

# Introduction
Window detection is a key component in many graphics and vision applications related to 3D city modeling and scene visualization. We present a novel approach for learning to recognize windows in a colored facade image. Rather than predicting bounding boxes or performing facade segmentation, our system locates keypoints of windows, and learns keypoint relationships to group them together into windows. A further module  provides extra recognizable information at the window center. Locations and relationships of keypoints are encoded in different types of heatmaps, which are learned in an end-to-end network. 
We have also constructed a facade dataset with 3418 annotated images  to facilitate  research in this field. It has richly varying facade structure, occlusion, lighting conditions, and angle of view.
On our dataset, our method achieves  precision of 91.4\% and  recall of 91.0\% under 50\% IoU. We also make a quantitative comparison with state-of-the-art methods to verify the utility of our proposed method. 
Applications based on our window detector are also demonstrated, such as window blending.

![image](https://github.com/lck1201/win_det_heatmaps/raw/master/docs/framework.jpg)

# Preparation
**Environment**

Please install PyTorch following the [official webite](https://pytorch.org/). In addition, you have to install other necessary dependencies.
```bash
pip3 install -r requirements.txt
```

**Dataset**

The zju_facade_jcst2020 database is described in the paper, and now avaliable on [BaiduYun](https://pan.baidu.com/s/1yfPmH7KZ4X5RoYWUhE9tYw)(code: qlx5), [GoogleDrive](https://drive.google.com/drive/folders/1TfeIcQ8KlEvP1-ewGcTaj3SqU_IpoLUv?usp=sharing)

Facade images were collected from the Internet and existing datasets including [TSG-20](http://dib.joanneum.at/cape/TSG-20/), [TSG-60](http://dib.joanneum.at/cape/TSG-60/), [ZuBuD](http://www.vision.ee.ethz.ch/showroom/zubud/), [CMP](http://cmp.felk.cvut.cz/~tylecr1/facade/), [ECP](http://vision.mas.ecp.fr/Personnel/teboul/data.php), and then data cleaning proceeded to ensure data quality standards. Using the open source software [LabelMe](https://github.com/wkentaro/labelme), we manually annotated the positions of four corners of windows in order.

<div align="center">
    <img src="https://github.com/lck1201/win_det_heatmaps/raw/master/docs/anno_example.jpg" width="550">
</div>

**Model**

You can use our trained models from [BaiduYun](https://pan.baidu.com/s/1l3Nsy3opmXQdcUgnAIY4rw)(code: n0ev), [GoogleDrive](https://drive.google.com/drive/folders/1TfeIcQ8KlEvP1-ewGcTaj3SqU_IpoLUv?usp=sharing). ResNet18, MobileNetV2, ShuffleNetV2 are provided. All the configurations are written in \*.yaml files and config_pytorch.py, and you can change it up to your own needs.

The table concludes the performance of three models on our i7-6700K + 1080Ti platform. Note that center verification module is not used.

| Architecture        | #Params | FLOPs | Time | P_50  | P_75  | P_mean | R_50  | R_75  | R_mean |
|---------------------|---------|-------|------|-------|-------|--------|-------|-------|--------|
| ShuffleNetV2 + Head | 13.8M   | 29.5G | 62ms | 85.2% | 62.8% | 54.9%  | 86.2% | 63.5% | 55.5%  |
| MobileNetV2 + Head  | 16.9M   | 31.2G | 65ms | 87.0% | 64.9% | 56.8%  | 90.0% | 67.1% | 58.5%  |
| ResNet18 + Head     | 19.6M   | 32.0G | 62ms | 88.4% | 68.4% | 58.7%  | 91.2% | 70.5% | 60.5%  |

# Usage
**Train**
```bash 
python train.py --cfg /path/to/yaml/config \
    --data /path/to/data/root \
    --out /path/to/output/root
```

**Test**
```bash 
python test.py --cfg /path/to/yaml/config --model /path/to/model \
    --data /path/to/data/root \
    --out /path/to/output/root
```


**Inference**
```bash
python infer.py --cfg /path/to/yaml/config \
                --model /path/to/model \
                --infer /path/to/image/directory
```

# Examples
<div align="center">
    <img src="https://github.com/lck1201/win_det_heatmaps/raw/master/docs/example.jpg" width="850">
</div>

# Applications
**Facade Unification**

We have developed a computational workflow for window texture blending based on our window detection method. Based on our technique, graphics designer can easily manipulate facade photos to create ideal building textures, while removing windows which are unsatisfactory  due to their open or closed status, lighting conditions and occlusion,  replacing them with the selected unified window texture.
<div align="center">
    <img src="https://github.com/lck1201/win_det_heatmaps/raw/master/docs/textureBlendingApp.jpg" width="650">
</div>

**Facade Beautification**

Applying the above workflow, image beautification can be also performed to generate visually pleasant results with mixed features. 
<div align="center">
    <img src="https://github.com/lck1201/win_det_heatmaps/raw/master/docs/facade_beautification.jpg" width="550">
</div>

**Facade Analytics**

As our method can efficiently locate windows in urban facade images, it is of use for automatically analyzing semantic structure and extracting numerical information. With additional simple steps, it is easy to determine the windows in a single row or  column. Furthermore, it can be adopted to predict building layers and symmetric feature lines.
<div align="center">
    <img src="https://github.com/lck1201/win_det_heatmaps/raw/master/docs/facade_analytics.jpg" width="650">
</div>

# Citation
If our code/dataset/models/paper helps your research, please cite with:
```bitex
@article{Chuan-Kang Li:900, 
    author = {Chuan-Kang Li, Hong-Xin Zhang, Jia-Xin Liu, Yuan-Qing Zhang, Shan-Chen Zou, Yu-Tong Fang},
    title = {Window Detection in Facades Using Heatmap Fusion},
    publisher = {Journal of Computer Science and Technology},
    year = {2020},
    journal = {Journal of Computer Science and Technology},
    volume = {35},
    number = {4},
    eid = {900},
    numpages = {12},
    pages = {900},
    keywords = {facade parsing;window detection;keypoint localization},
    url = {http://jcst.ict.ac.cn/EN/abstract/article_2660.shtml},
    doi = {10.1007/s11390-020-0253-4}
}    
```

 
# Acknowledgement
The major contributors of this repository include [Chuankang Li](https://github.com/lck1201), [Yuanqing Zhang](https://github.com/yuanqing-zhang), [Shanchen Zou](https://github.com/Generior), and [Hongxin Zhang](https://person.zju.edu.cn/zhx).


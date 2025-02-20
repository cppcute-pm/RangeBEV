<p align="center" width="100%">
<a target="_blank"><img src="assets/overview.png" alt="RangeBEV" style="width: 100%; min-width: 200px; display: block; margin: auto;"></a>
</p>

# Range and Bird's Eye View Fused Cross-Modal Visual Place Recognition

## 1) introduction
This repository contains the code for our proposed method "RangeBEV". the link of the paper is [Range and Bird's Eye View Fused Cross-Modal Visual Place Recognition](https://arxiv.org/abs/2502.11742)

We propose an innovative initial retrieval + re-rank method that effectively combines information from range (or RGB) images and Bird's Eye View (BEV) images. Our approach relies solely on a computationally efficient global descriptor similarity search process to achieve re-ranking. Additionally, we introduce a novel similarity label supervision technique to maximize the utility of limited training data. 

Experimental results on the KITTI dataset demonstrate that our method significantly outperforms state-of-the-art approaches.

### 1、pipeline
![pipeline](./assets/pipeline.png)
### 2、quantitative results
![quantitative results](./assets/main_results.png)
### 3、qualitative results
![qualitative results](./assets/re_rank_vis.png)

## 2) Get started
### 1、Environment Setup
We use PyTorch and the MMSegmentation Library. We acknowledge their great contributions!
```bash
conda create -yn RangeBEV python=3.8
conda activate RangeBEV
pip install -r requirements.txt
```
if you encounter problems with installing [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), [faiss](https://github.com/facebookresearch/faiss), [open3d](https://github.com/isl-org/Open3D), [vision3d](https://github.com/qinzheng93/vision3d), etc., please refer to the official websites for help.

### 2、Download Datasets
#### KITTI Odomety dataset
You should firstly login in the [KITTI official website](https://www.cvlibs.net/datasets/kitti/index.php) and then download the odometry dataset. Download the "color", "velodyne laser data", "calibration files" and "ground truth poses" .zip files. Unzip them into a folder structure according to the official guide. Then manually create a folder named "my_tool" for further use.

#### SemanticKITTI dataset
Then you need to download the SemanticKITTI label data from the [official website](https://semantic-kitti.org/dataset.html), which will be used as the ground truth for training model on 11~21 sequences.

After this step, you should have the following folder structure in a data root folder:
```bash
KITTI/
├── dataset/
│   ├── poses/
│   ├── sequences/
├── my_tool/

semanticKITTI/
├── dataset/
│   ├── sequences/
```

#### Boreas dataset
Additionally, if you want to run the model on the Boreas dataset, you can download it from the [official website](https://www.boreas.utias.utoronto.ca/#/). The demanding sequences are in the `datasets/Boreas_dp/mv_boreas_minuse.py`and only the LiDAR and the Camera sensor data are required. We only do the single modal and RGB to pointcloud cross modal experiments, if you want to run our proposed method on the Boreas dataset, you can may need a further data preprocessing.

### 3、Prepare Models
Prepare the pretrained model weights for the RGB image branch from [here](https://github.com/open-mmlab/mmsegmentation/blob/b040e147adfa027bbc071b624bedf0ae84dfc922/configs/sem_fpn/README.md), the model weights needed is `fpn_r50_512x512_160k_ade20k_20200718_131734-5b5a6ab9.pth` and you modified it first to change the keyname architecture in state_dict then save it as `/path/to/your/fpn_r50_512x512_160k_ade20k_20200718_131734-5b5a6ab9_v2.pth`.

Prepare the pretrained weights on KITTI for the monocular metric depth estimation model from [here](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth), the `depth_anything_metric_depth_outdoor.pt` is needed. Next pull the [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) repository and install according to the official guide. After that, you should move the `datasets/Kitti_Odometry_dp/generate_bev_v2_sobel.py` and `datasets/Kitti_Odometry_dp/my_pykitti_odometry.py` files to `yourfolder/Depth-Anything/metric_depth/` folder, at the same time, you should assign the pretrained weights to the right path.

### 4、Data Preprocessing
run the following command in order to preprocess the data.
```bash
cd datasets/Kitti_Odometry_dp
python rgb_image_pre_process_v8.py
python pointcloud_pre_process_v1.py
python generate_range_image.py
python generate_bev_v2_pc_nonground.py
python generate_UTM_coords_v1.py
cd ../../../Depth-Anything/metric_depth
python generate_bev_v2_sobel.py
```

## 3) Training
assign the `script.sh` correctly, you should set the `data_path`, `weight_path`, `config_path`, `need_eval`, `seed`, `local_rank` parameters, `seed` is fixed to 3407, `config_path` is `configs/model_5/phase_5_standard.py`. Then run the following command to train the model.
```bash
bash script.sh
```

## 4) Inference
you can train from scratch with your own model weights or just use our [pretrained model weights](https://pan.baidu.com/s/1i4X6gddc5PG7mleV0znl5w?pwd=6emw), run the following command to evaluate the model.
```bash
python Kitti_evaluate.py
```

## 5) Citation
If you find this work useful in your research, please consider citing:
```bibtex
@misc{peng2025rangebirdseyeview,
      title={Range and Bird's Eye View Fused Cross-Modal Visual Place Recognition}, 
      author={Jianyi Peng and Fan Lu and Bin Li and Yuan Huang and Sanqing Qu and Guang Chen},
      year={2025},
      eprint={2502.11742},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.11742}, 
}
```
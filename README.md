# F2P-Net
![Fig  1](https://github.com/user-attachments/assets/b46fff39-3569-4dec-9b0a-9af36c0694ce)
The code of 《F2P-Net: A Hybrid Prompt-Enhanced Dual-Branch Cooperative Network 
for Industrial Defect Segmentation with Limited Data》 will be continuously updated.....


## Abstract
In industrial surface defect detection, insufficient defective samples constitute a primary limiting factor for model performance. Traditional defect segmentation methods not only require extensive annotated defect data but also demonstrate limited capability in discriminating defect boundary regions, thereby limiting their applicability to modern industrial inspection systems. To address these challenges, we propose F2P-Net, a few-sample, highly precise industrial surface defect segmentation framework composed of three core modules. ViCNet (ViT and CNN collaborative encoder network) is a hierarchical feature encoder that synergistically integrates a primary vision transformer backbone with an auxiliary convolutional branch, thereby inheriting robust segmentation priors from large-scale vision models while enhancing sensitivity to microscopic defects. AFDec (automated geometric prompt and multi-scale feature fusion decoder) serves as the decoding component, and it employs automated geometric prompts to target defect regions and fuses multi-scale features across hierarchical levels, ensuring precise boundary delineation. EVPT (edge-enhanced visual prompt tuning) is a fine-tuning module incorporating edge-explicit visual prompt (EVP)to facilitate effective industrial domain adaptation of large vision models. The proposed method achieves considerable performance over existing full-data training approaches in metrics including mAP, Recall, and IoU using only 1.76%~3.06% of training images across NEU-Seg, MT, KolektorSDD2, and DAGM2007 datasets. Under full-data training, it attains state-of-the-art segmentation accuracies with IoU scores of 86.03%, 92.57%, 78.77%, and 82.55%, respectively. The network provides a novel solution for industrial applications with few-sample, high-precision defect segmentation.


## Installation
Please clone our project to your local machine and prepare our environment by the following commands:

We recommend you use the following environment installation command directly.
```
conda env create -f environment.yml
```
Alternatively, you may use the following environment installation command:
```
$: conda env create -f environment.yml
$: cd F2P-Net
$: conda create -n F2P-Net python=3.9
$: conda activate F2P-Net
$: python -m pip install -r requirements.txt
(F2P-Net) $: python -m pip install -e .
```

The code has been tested on NVIDAI RTX 4090 with Python 3.9, CUDA 11.7 and Pytorch 1.13.1. Any other devices and environments may require to update the code for compatibility.


## Data
Please refer to the README.md in the dataset-specific folders under `./data` to prepare each of them.

## Train
Before training, please download the SAM checkpoints to `./pretrained` from [here](https://github.com/facebookresearch/segment-anything#model-checkpoints).
After downloading, there should be:
```
F2P_Net/
    pretrained/
        sam_vit_b_01ec64.pth
        sam_vit_h_4b8939.pth
        sam_vit_l_0b3195.pth
    ...
```

For few-shot training, please run:
```
$: cd F2P-Net
$: pwd
/your_dir/F2P-Net
$: conda activate F2P-Net
(cat-sam) $: python train.py --dataset <your-target-dataset> --shot_num 64
If the argument `--shot_num` is not specified, training will proceed with the full-shot condition. 
```
Please prepare the following GPUs according to the following conditions:
 `--dataset`: 4 x NVIDIA RTX 4090 (24GB) or ones with similar memories

After running, the checkpoint with the best performance will be automatically saved to `./exp/{dataset}_{sam_type}_64shot`.


## Test

We provide the checkpoints for the 64-shot experiments.
To download the checkpoints, please visit [here](https://drive.google.com/drive/folders/1JEAJn7svhzPNcYtH9faejTumm6sqUzv0).

For testing, please run:
```
$: cd F2P-Net
$: pwd
/your_dir/F2P-Net
$: conda activate F2P-Net
(F2P-Net) $: python test.py --dataset <your-target-dataset> --ckpt_path <your-target-ckpt>
```


## Inference
For inference, please run:
```
$: cd F2P-Net
$: pwd
/your_dir/F2P-Net
$: conda activate F2P-Net
(F2P-Net) $: python inference.py --dataset <your-target-dataset> --ckpt_path <your-target-ckpt>
```


## Acknowledgement
We acknowledge the use of the following public resources throughout this work: [Segment Anything Model](https://github.com/facebookresearch/segment-anything), [CAT-SAM](https://github.com/weihao1115/cat-sam).

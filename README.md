# PSTUDA
## One-to-Multiple: A Progressive Style Transfer Unsupervised Domain-Adaptive Framework for Kidney Tumor Segmentation

| **[1 Introduction](#introduction)** 
| **[2 Requirements](#requirements)**
| **[3 Usage](#usage)**
| **[4 Citation](#citation)**
| **[5 Acknowledgments](#acknowledgments)** |

<a id="introduction"></a>
## Introduction

Official code for NeurIPS 2024 paper "[One-to-Multiple: A Progressive Style Transfer Unsupervised Domain-Adaptive Framework for Kidney Tumor Segmentation]()".

> In multi-sequence Magnetic Resonance Imaging (MRI), accurately segmenting the kidney and tumor using traditional supervised methods requires detailed, labor-intensive annotations for each sequence. Unsupervised Domain Adaptation (UDA) reduces this burden by aligning cross-modal features and addressing inter-domain differences. However, most UDA methods focus on one-to-one domain adaptation, limiting efficiency in multi-target scenarios. To address this challenge, we propose a novel and efficient One-to-Multiple Progressive Style Transfer Unsupervised Domain-Adaptive (PSTUDA) framework.

<div align=center><img src="PSTUDA.png", width="90%"></div>

<a id="requirements"></a>
## Requirements

Clone this repository:

```bash
git clone https://github.com/MLMIP/PSTUDA.git
cd PSTUDA/
```

Install the dependencies:

```bash
conda create -n PSTUDA python=3.7.13
conda activate PSTUDA
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

<a id="usage"></a>

## Usage

### Dataset Storage Format

The directory structure of our private MSKT dataset is as follows:

```
dataset/
├── data/
│   ├── case_000/
│   │   ├── t1c/
│   │   │   ├── img/
│   │   │   │   ├── img_slice008.npy
│   │   │   │   ├── img_slice009.npy
│   │   │   │   └── ...  # other slice files
│   │   │   ├── label/
│   │   │   │   ├── label_slice008.npy
│   │   │   │   ├── label_slice009.npy
│   │   │   │   └── ...  # other label files
│   │   ├── fst2w/
│   │   │   ├── img/
│   │   │   │   ├── img_slice005.npy
│   │   │   │   ├── img_slice006.npy
│   │   │   │   └── ...  # other slice files
│   │   │   ├── label/
│   │   │   │   ├── label_slice005.npy
│   │   │   │   ├── label_slice006.npy
│   │   │   │   └── ...  # other label files
│   │   ├── t2w/
│   │   │   └── ...  # structure is similar to t1c and fst2w
│   │   ├── dwi/
│   │   │   └── ...  # structure is similar to t1c and fst2w
│   ├── case_001/
│   │   └── ...  # other folder structures are similar
│   ├── case_002/
│   │   └── ...
│   └── ...
├── filter_t1c_train.txt
├── filter_fst2w_train.txt
├── filter_t2w_train.txt
├── filter_dwi_train.txt
└── test_t1c_slices.txt
```

Description:

The four `filter_*.txt` files index all slice files from different sequences in the `data/` directory (with filter indicating that only slices containing the target region are recorded). 
The `test_*_slices.txt` files index all slices in the source domain test set within the `data/` directory.
For example, each line in `filter_t1c_train.txt` records paths like `data/case_046/t1c/img/img_slice005.npy`, `data/case_046/t1c/img/img_slice006.npy`, and so on.

Though the private dataset is not publicly available, the provided dataset storage format is intended to support implementation with custom datasets.

### Running PSTUDA

If you prepare your custom data following the above storage format, you can start training by executing the following command in the terminal, and the results will be saved in the `output_dir`.


```bash
python train.py [img_size] [num_domains] [input_channel] [train_img_dir] [val_img_dir]
```

We provide a simple local visualization script to conveniently view generated results, adversarial loss, and other visualization information during training. By executing the following commands and entering ```http://localhost:1998/``` in your browser, you can access the local visualization interface.

```bash
cd util
python WebVision.py
```

<a id="citation"></a>

## Citation

If you find our work is useful for your research, please consider citing:

```
@article{hu2024one,
  title={One-to-Multiple: A Progressive Style Transfer Unsupervised Domain-Adaptive Framework for Kidney Tumor Segmentation},
  author={Hu, Kai and Li, Jinhao and Zhang, Yuan and Ye, Xiongjun and Gao, Xieping},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```

<a id="acknowledgments"></a>

## Acknowledgments

Our code is inspired by [StarGAN v2](https://github.com/clovaai/stargan-v2) and [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

# Calibrating LLMs with Label Smoothing

## Introduction

This is the official repository for the supporting code to our paper __Calibrated Language Models and How to Find Them with Label Smoothing__, presented at ICML 2025.

This repository is built off of two public repositories

- [open-instruct](https://github.com/allenai/open-instruct)
- [ml-cross-entropy](https://github.com/apple/ml-cross-entropy)

Each of these is included in their own separate folder and contains their own `requirements.txt` file for running. If you run into any issues during installation or running the code, please access the specific repositories.

## FAQ

Here are some issues that we ran into that may be helpful.

### Installing `flash-attn` is slow.

We ran into this and our solution was to install the package directly from the wheels [here](https://github.com/Dao-AILab/flash-attention/releases). Just match your `torch`, `gcc` and CUDA versions. We generally use the `abiFalse` version of any wheel.

### A specific tokenizer does not work when training models.

You may have to add some handling code in `open-instruct/open_instruct/dataset_transformation.py` if your tokenizer isn't directly supported.

### CUDA Out of Memory

We generally suggest to use at least 4 NVIDIA A100 80GB for training models. For testing/benchmarking, only a single 80GB GPU is necessary, but this can vary depending on the model (Gemma2 does not use `flash-attention` and therefore may require more resources).

#### Gemma models run out of memory more quickly.

Gemma models (from HuggingFace, as supported by `open-instruct`) use an `eager` attention implementation as opposed to `flash-attn`, thus these models usually consume more memory when training or used for inference. We suggest usually doubling the available memory compared to models that use `flash-attn` in these cases (ex. use 8 GPUs instead of 4 if a model using `flash-attn` requires at least this many for training). In our own case, we only had 8 GPUs available thus were not able to train beyond the 2B Gemma models.

## Citation

```
@inproceedings{
    huang2025calibrated,
    title={Calibrated Language Models and How to Find Them with Label Smoothing},
    author={Jerry Huang and Peng Lu and QIUHAO Zeng},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=soLNj4l2EL}
}
```

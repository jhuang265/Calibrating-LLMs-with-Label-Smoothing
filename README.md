# Calibrating LLMs with Label Smoothing

## Introduction

This is the official repository for the supporting code to our paper Calibrated Language Models and How to Find Them with Label Smoothing, to be presented at ICML 2025.

This repository is built off of two public repositories

- [open-instruct](https://github.com/allenai/open-instruct)
- [ml-cross-entropy](https://github.com/apple/ml-cross-entropy)

Each of these is included in their own separate folder and contains their own `requirements.txt` file for running. If you run into any issues during installation or running the code, please access the specific repositories.

## FAQ

Here are some issues that we ran into that may be helpful

### Installing `flash-attn` is slow.

We ran into this and our solution was to install the package directly from the wheels [here](https://github.com/Dao-AILab/flash-attention/releases). Just match your `torch`, `gcc` and CUDA versions. We generally use the `abiFalse` version of any wheel.

### A specific tokenizer does not work when training models.

You may have to add some handling code in `open-instruct/open_instruct/dataset_transformation.py` if your tokenizer isn't directly supported.

## Citation

To be added upon publication of proceedings.

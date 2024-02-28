# Why Transformers Need Adam: A Hessian Perspective
This repository contains PyTorch implementations of blockwise and full Hessian spectrum estimation of large-scale neural nets via the Stochastic Lanscoz Quadrature method.  Check out more descriptions in the paper https://arxiv.org/abs/2402.16788.

## How to use 

Our code for spectrum estimation could be easily plugged into most public implementations for neural-net training. Here we provide two examples.

### For vision models 

Set up the Python environment using Anaconda with the provided `vision_models/environment.yml` file.

```
conda env create -f vision_models/environment.yml
conda activate cvmodels
```

Run the code for hessian spectrum estimation. 

```
bash vision_models/run.sh
```

### For language models 

Set up the Python environment using Anaconda with the provided `language_models/environment.yml` file.

```
conda env create -f vision_models/environment.yml
conda activate gpt2
```

Run the code for hessian spectrum estimation. 

```
bash language_models/run_gpt2.sh
```



## Acknowledgements

The above code is heavily based on the code base of [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) and [NanoGPT](https://github.com/karpathy/nanoGPT/) . 

## Citation

If you find this code helpful, please cite our paper in the following format.

```
@article{zhang2024why,
  title     = {Why Transformers Need Adam: A Hessian Perspective},
  author    = {Zhang, Yushun and Congliang, Chen and Tian, Ding and Ziniu, Li and Sun, Ruoyu and Luo, Zhi-Quan},
  booktitle = {arXiv preprint arXiv:2402.16788},
  year      = {2024},
}
```

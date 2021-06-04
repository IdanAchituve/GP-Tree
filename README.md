# GP-Tree: A Gaussian Process Classifier for Few-Shot Incremental Learning
Gaussian processes (GPs) are non-parametric, flexible, models that work well in many tasks. Combining GPs with deep learning methods via deep kernel learning (DKL) is especially compelling due to the strong representational power induced by the network. However, inference in GPs, whether with or without DKL, can be computationally challenging on large datasets. For this purpose, we proposed GP-Tree, a novel method for multi-class classification with Gaussian processes and DKL. We developed a tree-based hierarchical model in which each internal node of the tree fits a GP to the data using the Pólya-Gamma augmentation scheme. As a result, our method scales well with both the number of classes and data size. We demonstrated our method effectiveness against other Gaussian process training baselines, and we showed how our general GP approach is easily applied to incremental few-shot learning and achieves state-of-the-art performance.

### Instructions
install this repo via
```bash
pip install -e .
```

### Download data:
1. Download the CUB-200-2011 dataset from: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
2. Place the data under ./dataset

### Run code:
```bash
cd FSCIL
python trainer.py
```

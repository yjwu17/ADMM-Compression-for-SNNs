# ADMM-Compression-for-SNNs
- Our code will be coming soon.
- A comprehensive compression method incorporating prune synapses, quantizing weight and activity sparcity for SNNs.
- Based on high-performance supervised training algorithm for SNNs named STBP and alternating direction method of multipliers (ADMM).

## Requirments: python 3.5

If you want to train/sparse/quantize spiking model, plz click the main file defined in corresponding train/sparse/quantize subfiles and modify the path of datasets.

## Subfile:
    1. train: train ANN/SNN model based on LeNet-5 model.
    2. quantization: load the pre-trained model from 'train' subfile, and quantize weights by ADAMM
    3. sparse: load the pre-trained model from 'train' subfile, and sparse weights by ADAMM
    4. activity_regularization: under construction
    
    

torch_prune_utility.py has prune configuration and functions for pruning.


# Reference
1. Wu Y, Deng L, Li G, et al. Spatio-temporal backpropagation for training high-performance spiking neural networks[J]. Frontiers in neuroscience, 2018, 12: 331.
2. Deng L*, Wu Y*(co-first), Hu Y, et al. Comprehensive snn compression using admm optimization and activity regularization[J]. IEEE Transactions on Neural Networks and Learning Systems, 2021.

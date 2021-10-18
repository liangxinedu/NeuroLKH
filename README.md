# NeuroLKH: Combining Deep Learning Model with Lin-Kernighan-Helsgaun Heuristic for Solving the Traveling Salesman Problem
Liang Xin, Wen Song, Zhiguang Cao, Jie Zhang. NeuroLKH: Combining Deep Learning Model with Lin-Kernighan-Helsgaun Heuristic for Solving the Traveling Salesman Problem, 35th Conference on Neural Information Processing Systems (NeurIPS), 2021. [[pdf]](https://arxiv.org/pdf/2110.07983.pdf)

Please cite our paper if this code is useful for your work.
```
@inproceedings{xin2021neurolkh,
    author = {Xin, Liang and Song, Wen and Cao, Zhiguang and Zhang, Jie},
    booktitle = {Advances in Neural Information Processing Systems},
    title = {NeuroLKH: Combining Deep Learning Model with Lin-Kernighan-Helsgaun Heuristic for Solving the Traveling Salesman Problem},
    volume = {34},
    year = {2021}
}
```

## Quick start
To connect the deep learning model Sparse Graph Network (Python) and the Lin-Kernighan-Helsgaun Heuristic (C Programming), we implement two versions.
* subprocess version. This version requires writting and reading data files to connect the two programming languages. To compile and test with our pretrained models for TSP instances with 100 nodes:
```bash
make
python data_generate.py -test
python test.py --dataset test/100.pkl --model_path pretrained/neurolkh.pt --n_samples 1000 --lkh_trials 1000 --neurolkh_trials 1000
```
* Swig (http://www.swig.org) version. The C code is wrapped for Python. To compile and test with our pretained models for TSP instances with 100 nodes:
```bash
bash setup.sh
python data_generate.py -test
python swig_test.py --dataset test/100.pkl --model_path pretrained/neurolkh.pt --n_samples 1000 --lkh_trials 1000 --neurolkh_trials 1000
```

## Usage
### Generate the training dataset
As the training for edge scores requires the edge labels, generating the training dataset will take a relatively long time (a couple of days).
```bash
python data_generate.py -train
```
### Train the NeuroLKH Model 
To train for the node penalties in the Sparse Graph Network, swig is required and the subprocess version is currently not supported. With one RTX 2080Ti GPU, the model converges in approximately 4 days.
```bash
CUDA_VISIBLE_DEVICES="0" python train.py --file_path train --eval_file_path val --eval_batch_size=100 --save_dir=saved/tsp_neurolkh --learning_rate=0.0001
```
### Finetune the node decoder for large sizes
The finetuning process takes less than 1 minute for each size.
```bash
CUDA_VISIBLE_DEVICES="0" python finetune_node.py
```
### Testing
Test with the pretrained model on TSP with 500 nodes:
```bash
python test.py --dataset test/500.pkl --model_path pretrained/neurolkh.pt --n_samples 1000 --lkh_trials 1000 --neurolkh_trials 1000
```
We test on the TSPLIB instances with two NeuroLKH Models, NeuroLKH trained with uniformly distributed TSP instances and NeuroLKH_M trained with uniform, clustered and uniform-clustered instances (please refer to the paper for details).
```bash
python tsplib_test.py
```

## Other Routing Problems (CVRP, PDP, CVRPTW)
### Testing with pretrained models
test for CVRP with 100 customers, PDP and CVRPTW with 40 customers
```bash
# Capacitated Vehicle Routing Problem (CVRP)
python CVRPdata_generate.py -test
python CVRP_test.py --dataset CVRP_test/cvrp_100.pkl --model_path pretrained/cvrp_neurolkh.pt --n_samples 1000 --lkh_trials 10000 --neurolkh_trials 10000
# Pickup and Delivery Problem (PDP)
python PDPdata_generate.py -test
python PDP_test.py --dataset PDP_test/pdp_40.pkl --model_path pretrained/pdp_neurolkh.pt --n_samples 1000 --lkh_trials 10000 --neurolkh_trials 10000
# CVRP with Time Windows (CVRPTW)
python CVRPTWdata_generate.py -test
python CVRPTw_test.py --dataset CVRPTW_test/cvrptw_40.pkl --model_path pretrained/cvrptw_neurolkh.pt --n_samples 1000 --lkh_trials 10000 --neurolkh_trials 10000
```
### Training
train for CVRP with 100-500 customers, PDP and CVRPTW with 40-200 customers
```bash
# Capacitated Vehicle Routing Problem (CVRP)
python CVRPdata_generate.py -train
CUDA_VISIBLE_DEVICES="0" python CVRP_train.py --save_dir=saved/cvrp_neurolkh
# Pickup and Delivery Problem (PDP)
python PDPdata_generate.py -train
CUDA_VISIBLE_DEVICES="0" python PDP_train.py --save_dir=saved/pdp_neurolkh
# CVRP with Time Windows (CVRPTW)
python CVRPTWdata_generate.py -train
CUDA_VISIBLE_DEVICES="0" python CVRPTW_train.py --save_dir=saved/cvrptw_neurolkh
```

## Dependencies
* Python >= 3.6
* Pytorch
* sklearn
* Numpy
* tqdm
* (Swig, optional)

## Acknowledgements
* The LKH code is the 3.0.6 version (http://www.akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.6.tgz)
* The code for Sparse Graph Network is build on the code framework of 'An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem' (INFORMS Annual Meeting Session 2019) (https://github.com/chaitjo/graph-convnet-tsp)

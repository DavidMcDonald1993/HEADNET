# HEADNet: Hyperbolic Embedding of Attributed Directed Networks
Reference implementation of HEADNet algorithm 

Authors: David MCDONALD (davemcdonald93@gmail.com) and Shan HE (s.he@cs.bham.ac.uk)

# Requirements:

* Python3
* Numpy
* Scipy
* Scikit-learn
* Scikit-multilearn
* Keras

## Required packages (pip)
Install the required packages with 
```bash
pip install -r requirements.txt 
```
## Setup environment (conda) 
The conda environment is described in headnet_env.yml.

Run 
```bash
conda env create -f headnet_env.yml
```
to create the environment, and 
```bsh
conda activate headnet-env
```
to activate it. 


# How to use:
Run the code with 
```bash
python main.py --graph path/to/graph.npz --features path/to/features.csv --embedding path/to/save/embedding -e *num_epochs*
```
Additional options can be viewed with 
```bash
python main.py --help
```


# Input Data Format
## Graph
Graphs are given as sparse adjacency matrices

## Node attributes and labels
labels and features should be comma separated tables indexed by node id

# Citation:
If you find this useful, please use the following citation (under review)
```
@article{mcdonald2020headnet,
  title={HEADNet: Hyperbolic Embedding of Attributed Directed Networks},
  author={McDonald, David and He, Shan},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2020},
  publisher={IEEE}
}

```
import numpy as np

import networkx as nx 

import itertools

import os

from scipy.sparse import load_npz, save_npz

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int)
    parser.add_argument("--dir", type=str)

    return parser.parse_args()

def main():

    args = parse_args()

    d = args.dir
    dataset = "gplus"
    seed = args.seed

    graph_filename = os.path.join("datasets",
        dataset, "graph.npz")
    assert os.path.exists(graph_filename)
    print ("reading graph from", graph_filename)
    graph = load_npz(graph_filename)

    removed_dir = os.path.join(d,
        dataset, "seed={:03d}".format(seed))

    output_filename = os.path.join(removed_dir,
        "training_edges", "graph.npz")

    if os.path.exists(output_filename):
        print (output_filename, "already exists")
        return

    removed_edges_filename = os.path.join(removed_dir, 
        "removed_edges", "test_edges.tsv")
    print ("reading removed edges from",
        removed_edges_filename)
    assert os.path.exists(removed_edges_filename)


    with open(removed_edges_filename, "r") as f:
        for line in (line.rstrip() 
            for line in f.readlines()):
                u, v = line.split("\t")
                graph[int(u), int(v)] = 0
    graph.eliminate_zeros()

    print ("writing to", output_filename)
    save_npz(output_filename, graph)
    print ()


        
if __name__ == "__main__":
    main()

import os 
import glob 

import numpy as np
import networkx as nx 
import pandas as pd 

from scipy.sparse import csr_matrix, save_npz

import argparse 

from itertools import count
from collections import defaultdict as ddict

class MyDict(dict):
    
    def __init__(self):
        super(MyDict, self).__init__()
        self._count = count()
    
    def missing(self, key):
        self.update({key: self._count.__next__()})
        return self.get(key)
    
    __missing__ = missing

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-directory",
        default="datasets/twitter")

    parser.add_argument("--ego-nodes", nargs="+",)

    return parser.parse_args()

def main():

    args = parse_args()

    output_dir = args.data_directory
    input_dir = os.path.join(args.data_directory, 
        "original")
    print ("reading from", input_dir)
    print ("writing to", output_dir)
    assert os.path.exists(input_dir)
    assert os.path.exists(output_dir)

    ego_nodes = args.ego_nodes
    print ("ego nodes are", ego_nodes)

    edgelist_filename = os.path.join(input_dir, 
        "gplus_combined.txt.gz")
    print ("reading graph from", edgelist_filename)
    assert os.path.exists(edgelist_filename)
    graph = nx.read_edgelist(edgelist_filename, 
        create_using=nx.Graph(), 
        nodetype=int)

    nx.set_edge_attributes(graph, name="weight", values=1.)

    output_edgelist_filename = os.path.join(output_dir, 
        "edgelist.tsv.gz")
    print ("writing graph to", output_edgelist_filename)
    # nx.write_edgelist(graph, output_edgelist_filename, 
    #               delimiter="\t", data=["weight"])

    # sparse adjacency
    adj = nx.adjacency_matrix(graph, 
        nodelist=sorted(graph))
    adj_filename = os.path.join(output_dir, 
        "graph.npz")
    print ("writing adjacency matrix to",
        adj_filename)
    save_npz(adj_filename, adj)

    ### EGO memberships
    ego_memberships = {n: set() 
        for n in sorted(graph)}

    print ("reading edgelists for ego memberships")
    for ego_node in ego_nodes:
        print ("processing ego node", ego_node)
        edgelist = os.path.join(input_dir,
            "{}.edges".format(ego_node))
        print ("reading", edgelist)
        g = nx.read_edgelist(edgelist, nodetype=int, 
            create_using=nx.Graph())
        g.add_edges_from([(ego_node, u) for u in g])
        for u in g:
            ego_memberships[u].add(ego_node)
        print()


    ### FEATURES
    featname_map = MyDict()

    node_feats = {n: set() 
        for n in sorted(graph)}

    for ego_node in ego_nodes:
    
        print ("PROCESSING EGO NODE", ego_node)
        
        featnames_file = os.path.join(input_dir,
            "{}.featnames".format(ego_node))
        print ("reading featnames from", featnames_file)
        ego_featname_map = {}
        with open(featnames_file, "r") as f:
            for line in (line.rstrip() for line in f.readlines()):
                split = line.split(" ")
                feat_id = int(split[0])
                feat_name = " ".join(split[1:])
                feat_name_mapped = featname_map[feat_name]
                ego_featname_map.update({feat_id: feat_name_mapped})
                
        ego_node_feat_file = os.path.join(input_dir,
            "{}.egofeat".format(ego_node))
        print ("reading ego feats from", ego_node_feat_file)
        with open(ego_node_feat_file, "r") as f:
            for line in (line.rstrip() for line in f.readlines()):
                for i, feat in enumerate(
                    (int(n) for n in line.split(" "))):
                    if feat:
                        node_feats[ego_node].add(ego_featname_map[i])
            
        feat_file = os.path.join(input_dir,
            "{}.feat".format(ego_node))
        print ("reading node feats from", feat_file)
        with open(feat_file, "r") as f:
            for line in (line.rstrip() for line in f.readlines()):
                split = [int(n) for n in line.split(" ")]
                node = split[0]
                ego_memberships[node].add(ego_node)
                for i, feat in enumerate(split[1:]):
                    if feat:
                        node_feats[node].add(ego_featname_map[i])
        print ()

    feats = np.zeros((len(graph), len(featname_map)), 
        dtype=bool)
    for n, feats_ in node_feats.items():
        feats[n, list(feats_)] = 1

    featname_map_inv = {v:k 
        for k, v in featname_map.items()}

    feats_df = pd.DataFrame(feats, 
        index=sorted(graph), 
        columns=[featname_map_inv[i] 
            for i in range(feats.shape[1])])
    feats_filename = os.path.join(output_dir, 
        "feats.csv.gz")
    print ("writing feats to", feats_filename)
    # feats_df.to_csv(feats_filename)

    feats = csr_matrix(feats)

    feats_sparse_filename = os.path.join(output_dir, 
        "feats.npz")
    print ("writing features sparse to", feats_sparse_filename)
    save_npz(feats_sparse_filename, feats)

    ### CIRCLE memberships
    circle_memberships = {n: set() 
        for n in sorted(graph)}

    print ("reading circle memberships")
    for ego_node in ego_nodes:
        
        print ("PROCESSING EGO NODE", ego_node)
        
        ego_circle_filename = os.path.join(input_dir,
            "{}.circles".format(ego_node))
        print ("reading circles from", ego_circle_filename)
        with open(ego_circle_filename, "r") as f:
            for line in (line.rstrip() for line in f.readlines()):
                split = line.split("\t")
                circle = int(split[0].split("circle")[1])
                for node in (int(n) for n in split[1:]):
                    ego_memberships[node].add(ego_node)
    #                 assert ego_node in ego_memberships[node], node
                    circle_memberships[node].add(circle)
        print ()

    circles = sorted(set().union(*circle_memberships.values()))
    
    # make label dfs
    ego_memberships_df = pd.DataFrame(0, 
        index=sorted(graph), columns=ego_nodes)
    circle_memberships_df = pd.DataFrame(0, 
        index=sorted(graph), columns=circles)    

    for n, ego_mems in ego_memberships.items():
        for ego in ego_mems:
            ego_memberships_df.at[n, ego] = 1   

    for n, circle_mems in circle_memberships.items():
        for circle in circle_mems:
            circle_memberships_df.at[n, circle] = 1   
            
    ego_memberships_filename = os.path.join(output_dir,
        "ego_memberships.csv.gz")
    print ("writing ego memberships to", ego_memberships_filename)
    # ego_memberships_df.to_csv(ego_memberships_filename)

    circle_memberships_filename = os.path.join(output_dir,
        "circle_memberships.csv.gz")
    print ("writing circle memberships to", circle_memberships_filename)
    # circle_memberships_df.to_csv(circle_memberships_filename)


    # labels sparse
    ego_memberships = ego_memberships_df.values
    ego_memberships = csr_matrix(ego_memberships)

    circle_memberships = circle_memberships_df.values
    circle_memberships = csr_matrix(circle_memberships)

    ego_memberships_sparse_filename = os.path.join(output_dir,
        "ego_memberships.npz")
    print ("writing ego memberships sparse to",
        ego_memberships_sparse_filename)
    save_npz(ego_memberships_sparse_filename,
        ego_memberships)

    circle_memberships_sparse_filename = os.path.join(output_dir,
        "circle_memberships.npz")
    print ("writing circle memberships sparse to",
        circle_memberships_sparse_filename)
    save_npz(circle_memberships_sparse_filename, 
        circle_memberships)

if __name__ == "__main__":
    main()
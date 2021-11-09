import numpy as np 
import networkx as nx 

import os

from scipy.sparse import save_npz

def main():

    output_dir = os.path.join("datasets", 
        "synthetic_scale_free")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    N = 1000

    for seed in range(30):

        d = os.path.join(output_dir, "{:02d}".format(seed))
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

        g = nx.DiGraph(nx.scale_free_graph(N, seed=seed))
        print ("seed", seed, "number of nodes", len(g),
            "number of edges", len(g.edges))
        nx.set_edge_attributes(g, name="weight", values=1.)
       
        nx.write_edgelist(g, os.path.join(d, "edgelist.tsv.gz"), 
            delimiter="\t", data=["weight"])

        A = nx.adjacency_matrix(g, nodelist=sorted(g))
        save_npz(os.path.join(d, "graph.npz"), A)



if __name__ == "__main__":
    main()
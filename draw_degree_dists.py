import os 
import networkx as nx

from headnet.visualise import plot_degree_dist


def main():

    datasets = os.listdir("datasets")

    for dataset in datasets:

        if "synthetic" in dataset:
            continue

        edgelist = os.path.join("datasets", 
            dataset, "edgelist.tsv.gz")

        print ("reading edgelist", edgelist)
        graph = nx.read_weighted_edgelist(edgelist,
            create_using=nx.DiGraph())
        
        filename = os.path.join("",
            # "datasets", 
            # dataset, 
            "{}_degree_dist.png".format(dataset))
        plot_degree_dist(graph, dataset.capitalize(), filename)


if __name__ == "__main__":
    main()
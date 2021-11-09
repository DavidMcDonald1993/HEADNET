import os 

import pandas as pd 

from collections import defaultdict

def make_dir(dir):
    if not os.path.exists(dir):
        print ("making directory", dir)
        os.makedirs(dir, exist_ok=True)

def main():

    results_dir = os.path.join("collated_results", 
        "lp_experiment")
    assert os.path.exists(results_dir)

    summary_dir = os.path.join(results_dir, "summary_recommender")
    make_dir(summary_dir)

    unattributed_datasets = ["synthetic_scale_free", "wiki_vote"]
    attributed_datasets = ["cora_ml", "citeseer", "pubmed", "cora", ]

    dims = (5, 10, 25, 50)

    cols_of_interest = ["map_lp",]
    
    cols_renamed = ["mAP", ]

    unattributed_index = [
        "ln",
        "harmonic",
        "line",
        "g2g_k=01_nofeats",
        "g2g_k=03_nofeats",
        "NK",
        "HEDNet_identity",
        "HEDNet",
    ]
    attributed_index = [
        "g2g_k=01_feats",
        "g2g_k=03_feats",
        "HEADNet_identity",
        "HEADNet",
    ]

    index = unattributed_index + attributed_index
    index_renamed = {
        "ln": "ATP (log)",
        "harmonic": "ATP (harmonic)",
        "line": "LINE",
        "g2g_k=01_nofeats": "G2G$_{\\text{NA}, K=1}$",
        "g2g_k=03_nofeats": "G2G$_{\\text{NA}, K=3}$",
        "NK": "NK",
        "HEDNet_identity": "HEADNet$_{\\text{NA}, \\Sigma=\\mathbb{I}}$",
        "HEDNet": "HEADNet$_{\\text{NA}}$",
        "g2g_k=01_feats": "G2G$_{K=1}$",
        "g2g_k=03_feats": "G2G$_{K=3}$",
        "HEADNet_identity": "HEADNet$_{\\Sigma=\\mathbb{I}}$",
        "HEADNet": "HEADNet",
    }

    index_of_interest = [
        "HEADNet$_{\\Sigma=\\mathbb{I}}$",
        "HEADNet",
    ]
    
    output_df = pd.DataFrame()

    for dim in dims:

        ttest_dir = os.path.join(results_dir, "dim={:03d}".format(dim), "t-tests")
        assert os.path.exists(ttest_dir)

        dim_dfs = defaultdict(list)

        for dataset in attributed_datasets:

            for df in ("means", "stds"):

                filename = os.path.join(results_dir, "dim={:03d}".format(dim),
                    "{}_dim={:03d}_{}.csv".format(dataset, dim, df))
                print ("reading", filename)
                results_df = pd.read_csv(filename, 
                    index_col=0)
                
                results_df = results_df[cols_of_interest]
                results_df.columns = cols_renamed

                if dataset in unattributed_datasets:
                    assert results_df.shape[0] == len(unattributed_index)
                    # handle networks without attributes
                    results_df = results_df.append(
                        pd.DataFrame([["--"]*len(cols_renamed)]*len(attributed_index),
                        index=attributed_index, columns=cols_renamed)) 
            
                results_df.index = [index_renamed[idx] 
                    for idx in results_df.index]

                results_df = results_df.loc[index_of_interest]

                results_df.index = results_df.index.map(lambda idx: dataset+"-"+idx)

                dim_dfs[df].append(results_df)
       
            # ttest 
            headnet_alg = "HEADNet"
            best_benchmark = "HEADNet_identity"

            ttest_filename = os.path.join(ttest_dir, 
                "{}_dim={:03d}_ttest-{}-{}.csv".format(dataset, dim, 
                headnet_alg, best_benchmark))
            print ("reading ttests from", ttest_filename)
            ttest = pd.read_csv(ttest_filename, index_col=0)

            ttest = ttest[cols_of_interest]
            ttest.columns = cols_renamed

            # format algorithm rows
            # ttest.at[headnet_alg] = output_df.loc[index_renamed[headnet_alg]]#["{:.03f}".format(x) 
            # ttest.at[best_benchmark] = output_df.loc[index_renamed[best_benchmark]]#["{:.03f}".format(x) 
            ttest.at["t-statistic"] = ["{:.03e}".format(x) 
                for x in ttest.loc["t-statistic"]]
            ttest.at["p-value"] = ["{:.03e}".format(x) 
                for x in ttest.loc["p-value"]]
            ttest.at["rejected_null?"] = [int(x) 
                for x in ttest.loc["rejected_null?"]]

            ttest.index = [
                index_renamed[headnet_alg],
                index_renamed[best_benchmark], 
                "$t$-statistic",
                "$p$-value",
                "$p<0.05$"
            ]

            # put benchmark first
            ttest = ttest.reindex([
                index_renamed[best_benchmark], 
                index_renamed[headnet_alg],
                "$t$-statistic",
                "$p$-value",
                "$p<0.05$"
            ])

            ttest_filename = os.path.join(summary_dir, 
                f"{dim}-{dataset}-ttest.csv")
            print ("writing to", ttest_filename)
            ttest.to_csv(ttest_filename)


        dim_dfs["means"] = pd.concat(dim_dfs["means"])
        dim_dfs["stds"] = pd.concat(dim_dfs["stds"])

        mean_col = dim_dfs["means"]["mAP"]
        std_col = dim_dfs["stds"]["mAP"]

        output_df[dim] = pd.Series(["{:0.3f}({:.3f})".format(m, s) 
            if not isinstance(m, str) else m
                for m, s in zip(mean_col.values, std_col.values)],
                    index=mean_col.index)



    output_filename = os.path.join(summary_dir, 
        "recommender.csv")
    print ("writing to", output_filename)
    output_df.to_csv(output_filename)

    print ()


if __name__ == "__main__":
    main()
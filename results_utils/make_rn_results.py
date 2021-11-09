import os 

import pandas as pd 

def make_dir(dir):
    if not os.path.exists(dir):
        print ("making directory", dir)
        os.makedirs(dir, exist_ok=True)

def main():

    results_dir = os.path.join("collated_results", 
        "rn_experiment")
    assert os.path.exists(results_dir)

    summary_dir = os.path.join(results_dir, "summary")

    make_dir(summary_dir)

    unattributed_datasets = [
        # "synthetic_scale_free", "wiki_vote"
        ]
    attributed_datasets = ["cora_ml", "citeseer", "pubmed", "cora", ]

    dims = (5, 10, 25, 50)

    cols_of_interest = ["mean_rank_lp", "roc_lp", "ap_lp", 
        "map_lp", "p@1", "p@3", "p@5", "p@10",]
    
    cols_renamed = ["Mean Rank", "AUROC", "AP", 
        "mAP", "p@1", "p@3", "p@5", "p@10",]

    unattributed_index = [
        # "ln",
        # "harmonic",
        # "line",
        # "g2g_k=01_nofeats",
        # "g2g_k=03_nofeats",
        # "HEDNet",
    ]
    attributed_index = [
        "g2g_k=01_feats",
        "g2g_k=03_feats",
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

    unattributed_benchmark_algorithms = [alg for alg in unattributed_index 
        if "HEDNet" not in alg]
    attributed_benchmark_algorithms = [alg for alg in attributed_index 
        if "HEADNet" not in alg]

    # desired_index_order = ["ATP (log)", "ATP (harmonic)",
    #     "LINE", "G2G$_{K=1}$", "aG2G$_{K=1}$", 
    #      "G2G$_{K=3}$", "aG2G$_{K=3}$", 
    #      "HEDNet", "aHEDNet"]


    for dim in dims:

        ttest_dir = os.path.join(results_dir, "dim={:03d}".format(dim), "t-tests")
        assert os.path.exists(ttest_dir)

        output_dir = os.path.join(summary_dir, str(dim))
        make_dir(output_dir)

        for dataset in unattributed_datasets + attributed_datasets:

            dfs = {}

            for df in ("means", "stds"):

                filename = os.path.join(results_dir, "dim={:03d}".format(dim),
                    "{}_dim={:03d}_{}.csv".format(dataset, dim, df))
                print ("reading", filename)
                results_df = pd.read_csv(filename, 
                    index_col=0)
                
                results_df = results_df[cols_of_interest]
                results_df.columns = cols_renamed

                # if dataset in unattributed_datasets:
                #     assert results_df.shape[0] == len(unattributed_index)
                #     # handle networks without attributes
                #     results_df = results_df.append(
                #         pd.DataFrame([["--"]*len(cols_renamed)]*len(attributed_index),
                #         index=attributed_index, columns=cols_renamed)) 
                assert dataset in attributed_datasets
                benchmark_algorithms = attributed_benchmark_algorithms
                headnet_alg = "HEADNet"

                # select ttest
                if df == "means":
                    best_benchmark = results_df.loc[benchmark_algorithms]["AP"].astype(float).idxmax()
            
                results_df.index = [index_renamed[idx] 
                    for idx in results_df.index]
                
                dfs[df] = results_df

            output_df = pd.DataFrame()

            for col in cols_renamed:
                mean_col = dfs["means"][col]
                std_col = dfs["stds"][col]

                if "Rank" in col:
                    output_df[col] = pd.Series(["{:0.1f}({:.1f})".format(m, s) 
                        if not isinstance(m, str) else m
                        for m, s in zip(mean_col.values, std_col.values)],
                            index=mean_col.index)
                else:

                    output_df[col] = pd.Series(["{:0.3f}({:.3f})".format(m, s) 
                        if not isinstance(m, str) else m
                        for m, s in zip(mean_col.values, std_col.values)],
                            index=mean_col.index)

            output_filename = os.path.join(output_dir, 
                dataset + ".csv")
            print ("writing to", output_filename)
            output_df.to_csv(output_filename)

            ttest_filename = os.path.join(ttest_dir, 
                "{}_dim={:03d}_ttest-{}-{}.csv".format(dataset, dim, 
                headnet_alg, best_benchmark))
            print ("reading ttests from", ttest_filename)
            ttest = pd.read_csv(ttest_filename, index_col=0)

            ttest = ttest[cols_of_interest]
            ttest.columns = cols_renamed

            # format algorithm rows
            ttest.at[headnet_alg] = output_df.loc[index_renamed[headnet_alg]]#["{:.03f}".format(x) 
                # if "Rank" not in i else "{:.01f}".format(x) 
                # for i, x in ttest.loc[headnet_alg].items()]
            ttest.at[best_benchmark] = output_df.loc[index_renamed[best_benchmark]]#["{:.03f}".format(x) 
                # if "Rank" not in i else "{:.01f}".format(x) 
                # for i, x in ttest.loc[best_benchmark].items()]
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

            ttest_filename = os.path.join(output_dir, 
                dataset + "-ttest.csv")
            print ("writing to", ttest_filename)
            ttest.to_csv(ttest_filename)
        
            print ()


if __name__ == "__main__":
    main()
#!/bin/bash

#SBATCH --job-name=G2GCITLP
#SBATCH --output=G2GCITLP_%A_%a.out
#SBATCH --error=G2GCITLP_%A_%a.err
#SBATCH --array=0-1919
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=5G

scales=(False)
datasets=(cora_ml citeseer pubmed cora)
dims=(5 10 25 50)
seeds=({0..29})
ks=(01 03)
exp=lp_experiment
feats=(nofeats feats)

num_scales=${#scales[@]}
num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}
num_ks=${#ks[@]}
num_feats=${#feats[@]}

scale_id=$((SLURM_ARRAY_TASK_ID / (num_feats * num_ks * num_seeds * num_dims * num_datasets) % num_scales))
dataset_id=$((SLURM_ARRAY_TASK_ID / (num_feats * num_ks * num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / (num_feats * num_ks * num_seeds) % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID / (num_feats * num_ks) % num_seeds ))
k_id=$((SLURM_ARRAY_TASK_ID / num_feats % num_ks ))
feat_id=$((SLURM_ARRAY_TASK_ID % num_feats ))

scale=${scales[$scale_id]}
dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}
k=${ks[$k_id]}
feat=${feats[$feat_id]}

echo $scale $dataset $dim $seed $k $feat  

data_dir=datasets/${dataset}
graph=${data_dir}/graph.npz

removed_edges_dir=$(printf edgelists/${dataset}/seed=%03d/removed_edges ${seed})

embedding_dir=../graph2gauss/embeddings/${dataset}/${feat}/${exp}
embedding_dir=$(printf "${embedding_dir}/scale=${scale}/k=${k}/seed=%03d/dim=%03d/" ${seed} ${dim})

test_results=$(printf \
    "test_results/${dataset}/${exp}/dim=%03d/g2g_k=${k}_${feat}/" ${dim})

if [ ! -f ${test_results}/${seed}.pkl ]
then
    args=$(echo --graph ${graph} \
        --removed_edges_dir ${removed_edges_dir} \
        --dist_fn kle \
        --embedding ${embedding_dir} --seed ${seed} \
        --test-results-dir ${test_results})
    echo ${args}

    module purge
    module load bluebear
    module load future/0.16.0-foss-2018b-Python-3.6.6

    python evaluate_lp.py ${args}
else 
    echo ${test_results}/${seed}.pkl already exists 
fi
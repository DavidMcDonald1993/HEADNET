#!/bin/bash

#SBATCH --job-name=HEADNETRN
#SBATCH --output=HEADNETRN_%A_%a.out
#SBATCH --error=HEADNETRN_%A_%a.err
#SBATCH --array=0-479
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=5G

datasets=(cora_ml citeseer pubmed cora)
dims=(5 10 25 50)
seeds=({0..29})
exp=rn_experiment

num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / (num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / num_seeds % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID % num_seeds ))

dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}

data_dir=datasets/${dataset}
# graph=${data_dir}/edgelist.tsv.gz
graph=${data_dir}/graph.npz
embedding_dir=embeddings/${dataset}/${exp}
removed_edges_dir=$(printf nodes/${dataset}/seed=%03d/removed_edges ${seed})

test_results=$(printf \
    "test_results/${dataset}/${exp}/dim=%03d/HEADNet/" ${dim})
embedding_dir=$(printf \
    "${embedding_dir}/seed=%03d/dim=%03d" ${seed} ${dim})
echo ${embedding_dir}
echo ${test_results}

args=$(echo --graph ${graph} \
    --removed_edges_dir ${removed_edges_dir} \
    --dist_fn klh \
    --embedding ${embedding_dir} --seed ${seed} \
    --test-results-dir ${test_results})
echo ${args}

module purge
module load bluebear
# module load apps/python3/3.5.2
module load future/0.16.0-foss-2018b-Python-3.6.6

python evaluate_lp.py ${args}

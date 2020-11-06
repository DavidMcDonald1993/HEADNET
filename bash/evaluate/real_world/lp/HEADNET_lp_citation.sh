#!/bin/bash

#SBATCH --job-name=HEADNETCITLP
#SBATCH --output=HEADNETCITLP_%A_%a.out
#SBATCH --error=HEADNETCITLP_%A_%a.err
#SBATCH --array=0-959
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=5G

datasets=(cora_ml citeseer pubmed cora)
dims=(5 10 25 50)
seeds=({0..29})
exp=lp_experiment
feats=(nofeats feats)

num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}
num_feats=${#feats[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / (num_feats * num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / (num_feats * num_seeds) % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID / num_feats % num_seeds ))
feat_id=$((SLURM_ARRAY_TASK_ID % num_feats ))

dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}
feat=${feats[$feat_id]}

dataset=cora

echo $dataset $dim $seed $feat

data_dir=datasets/${dataset}
graph=${data_dir}/graph.npz

embedding_dir=embeddings/${dataset}/${feat}/${exp}
embedding_dir=$(printf \
    "${embedding_dir}/seed=%03d/dim=%03d" ${seed} ${dim})

removed_edges_dir=$(printf edgelists/${dataset}/seed=%03d/removed_edges ${seed})

test_results=$(printf \
    "test_results/${dataset}/${exp}/dim=%03d" ${dim})
if [ ${feat} == feats ]
then 
    test_results=${test_results}/HEADNet
else
    test_results=${test_results}/HEDNet
fi

if [ ! -f ${test_results}/${seed}.pkl ]
then
    args=$(echo --graph ${graph} \
        --removed_edges_dir ${removed_edges_dir} \
        --dist_fn klh \
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

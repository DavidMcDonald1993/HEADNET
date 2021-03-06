#!/bin/bash

#SBATCH --job-name=HEADNETWIKIRECON
#SBATCH --output=HEADNETWIKIRECON_%A_%a.out
#SBATCH --error=HEADNETWIKIRECON_%A_%a.err
#SBATCH --array=0-119
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=5G

datasets=(wiki_vote)
dims=(5 10 25 50)
seeds=({0..29})
exp=recon_experiment
feats=(nofeats)

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

echo $dataset $dim $seed $feat

data_dir=datasets/${dataset}
graph=${data_dir}/graph.npz
embedding_dir=embeddings/${dataset}/${feat}/${exp}/
embedding_dir=$(printf \
    "${embedding_dir}/seed=%03d/dim=%03d/" ${seed} ${dim})

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
    args=$(echo --graph ${graph} --dist_fn klh \
        --embedding ${embedding_dir} --seed ${seed} \
        --test-results-dir ${test_results})
    echo ${args}

    module purge
    module load bluebear
    module load future/0.16.0-foss-2018b-Python-3.6.6

    python evaluate_reconstruction.py ${args}
else
    echo ${test_results}/${seed}.pkl already exists
fi

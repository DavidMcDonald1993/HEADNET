#!/bin/bash

#SBATCH --job-name=NKSYNRECON
#SBATCH --output=NKSYNRECON_%A_%a.out
#SBATCH --error=NKSYNRECON_%A_%a.err
#SBATCH --array=0-119
#SBATCH --time=20:00
#SBATCH --ntasks=1
#SBATCH --mem=5G

datasets=({0..29})
dims=(5 10 25 50)
seeds=(0)
exp=recon_experiment
feat=nofeats

num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / (num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / num_seeds % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID % num_seeds ))

dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}

echo $dataset $dim $seed

data_dir=$(printf datasets/synthetic_scale_free/%02d ${dataset})
graph=${data_dir}/graph.npz

embedding_dir=$(printf \
    ../poincare-embeddings/embeddings/directed/synthetic_scale_free/%02d ${dataset})
embedding_dir=$(printf \
    "${embedding_dir}/dim=%02d/seed=%03d/${exp}" ${dim} ${seed})

test_results=$(printf \
    "test_results/synthetic_scale_free/${exp}/dim=%03d/NK" ${dim})

if [ ! -f ${test_results}/${dataset}.pkl ]
then

    args=$(echo --graph ${graph} --dist_fn poincare \
        --embedding ${embedding_dir} --seed ${dataset} \
        --test-results-dir ${test_results})
    echo ${args}

    module purge
    module load bluebear
    module load future/0.16.0-foss-2018b-Python-3.6.6

    python evaluate_reconstruction.py ${args}
else
     echo ${test_results}/${dataset}.pkl already exists 
fi

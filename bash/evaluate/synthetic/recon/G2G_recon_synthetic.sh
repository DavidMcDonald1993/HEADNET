#!/bin/bash

#SBATCH --job-name=G2GSYNRECON
#SBATCH --output=G2GSYNRECON_%A_%a.out
#SBATCH --error=G2GSYNRECON_%A_%a.err
#SBATCH --array=0-239
#SBATCH --time=20:00
#SBATCH --ntasks=1
#SBATCH --mem=5G

scales=(False)
datasets=({0..29})
dims=(5 10 25 50)
seeds=(0)
ks=(01 03)
exp=recon_experiment
feat=nofeats

num_scales=${#scales[@]}
num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}
num_ks=${#ks[@]}

scale_id=$((SLURM_ARRAY_TASK_ID / (num_ks * num_seeds * num_dims * num_datasets) % num_scales))
dataset_id=$((SLURM_ARRAY_TASK_ID / (num_ks * num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / (num_ks * num_seeds) % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID / num_ks % num_seeds ))
k_id=$((SLURM_ARRAY_TASK_ID % (num_ks) ))

scale=${scales[$scale_id]}
dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}
k=${ks[$k_id]}

echo $scale $dataset $dim $seed $k 

data_dir=$(printf datasets/synthetic_scale_free/%02d ${dataset})
graph=${data_dir}/graph.npz

embedding_dir=$(printf ../graph2gauss/embeddings/synthetic_scale_free/%02d/${exp}/scale=${scale}/k=${k} ${dataset})
embedding_dir=$(printf \
    "${embedding_dir}/seed=%03d/dim=%03d/" ${seed} ${dim})

test_results=$(printf \
    "test_results/synthetic_scale_free/${exp}/dim=%03d/g2g_k=${k}_${feat}" ${dim})

if [ ! -f ${test_results}/${dataset}.pkl ]
then

    args=$(echo --graph ${graph} --dist_fn kle \
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
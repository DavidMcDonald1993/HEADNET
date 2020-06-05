#!/bin/bash

#SBATCH --job-name=removeEdgesSynthetic
#SBATCH --output=removeEdgesSynthetic_%A_%a.out
#SBATCH --error=removeEdgesSynthetic_%A_%a.err
#SBATCH --array=0-29
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --mem=1G

datasets=({00..29})
seeds=(0)

num_datasets=${#datasets[@]}
num_seeds=${#seeds[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / num_seeds % num_datasets ))
seed_id=$((SLURM_ARRAY_TASK_ID % (num_seeds) ))

dataset=${datasets[$dataset_id]}
seed=${seeds[$seed_id]}

graph=datasets/synthetic_scale_free/${dataset}/graph.npz
output=edgelists/synthetic_scale_free/${dataset}

edgelist_f=$(printf "${output}/seed=%03d/training_edges/graph.npz" ${seed} )

if [ ! -f $edgelist_f  ]
then
	module purge
	module load bluebear
	module load future/0.16.0-foss-2018b-Python-3.6.6
	
	args=$(echo --graph=$graph \
		--output=$output --seed $seed)

	python remove_edges.py ${args}
fi
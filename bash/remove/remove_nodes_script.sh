#!/bin/bash

#SBATCH --job-name=removeNodes
#SBATCH --output=removeNodes_%A_%a.out
#SBATCH --error=removeNodes_%A_%a.err
#SBATCH --array=0-149
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --mem=15G

datasets=(cora_ml citeseer pubmed twitter gplus)
seeds=({0..29})

num_datasets=${#datasets[@]}
num_seeds=${#seeds[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / num_seeds % num_datasets ))
seed_id=$((SLURM_ARRAY_TASK_ID % (num_seeds) ))

dataset=${datasets[$dataset_id]}
seed=${seeds[$seed_id]}

echo $dataset $seed

edgelist=datasets/${dataset}/edgelist.tsv.gz
output=nodes/${dataset}

edgelist_f=$(printf "${output}/seed=%03d/training_edges/edgelist.tsv" ${seed} )

if [ ! -f $edgelist_f  ]
then
	module purge
	module load bluebear
	module load apps/python3/3.5.2

	args=$(echo "--edgelist=$edgelist \
		--output=$output --seed $seed")

	echo $args

	python remove_nodes.py ${args}

else
	echo $edgelist_f already exists
fi
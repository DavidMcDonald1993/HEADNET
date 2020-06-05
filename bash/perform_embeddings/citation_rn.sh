#!/bin/bash

#SBATCH --job-name=RNCITATION
#SBATCH --output=RNCITATION_%A_%a.out
#SBATCH --error=RNCITATION_%A_%a.err
#SBATCH --array=0-479
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=2
#SBATCH --mem=5G

e=1000

datasets=(cora_ml citeseer pubmed cora)
dims=(5 10 25 50)
seeds=({0..29})
exps=(rn_experiment)
feat=feats

num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}
num_exps=${#exps[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / (num_exps * num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / (num_exps * num_seeds) % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID / (num_exps) % num_seeds))
exp_id=$((SLURM_ARRAY_TASK_ID % num_exps))

dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}
exp=${exps[$exp_id]}

echo $dataset $dim $seed $exp $feat

data_dir=datasets/${dataset}
if [ $exp == "lp_experiment" ]
then 
    graph=$(printf edgelists/${dataset}/seed=%03d/training_edges/graph.npz ${seed})
elif [ $exp == "rn_experiment" ]
then 
    graph=$(printf nodes/${dataset}/seed=%03d/training_edges/graph.npz ${seed})
else 
    graph=${data_dir}/graph.npz
fi
echo graph is $graph
features=${data_dir}/feats.npz

embedding_dir=embeddings/${feat}/${dataset}/${exp}
embedding_dir=$(printf "${embedding_dir}/seed=%03d/dim=%03d" ${seed} ${dim})

echo embedding directory is $embedding_dir

if [ ! -f ${embedding_dir}/final_embedding.csv.gz ]
then 
    module purge
    module load bluebear

    if [ ! -f ${embedding_dir}/final_embedding.csv ]
    then 
        echo ${embedding_dir}/final_embedding.csv is missing -- performing embedding 

        module load TensorFlow/1.10.1-foss-2018b-Python-3.6.6
        pip install --user keras==2.2.4

        args=$(echo --graph ${graph} \
        --embedding ${embedding_dir} --seed ${seed} \
        --dim ${dim} --workers 1 -e ${e} \
        --nneg 10)
        if [ ${feat} == feats ]
        then
            args=${args}" --features ${features}"
        fi

        python main.py ${args}

    fi

    echo compressing ${embedding_dir}/final_embedding.csv
    gzip ${embedding_dir}/final_embedding.csv
    echo compressing ${embedding_dir}/final_variance.csv
    gzip ${embedding_dir}/final_variance.csv
else

    echo ${embedding_dir}/final_embedding.csv.gz already exists
    echo ${embedding_dir}/final_variance.csv.gz already exists
fi
#!/bin/bash

#SBATCH --job-name=HEADNET
#SBATCH --output=HEADNET_%A_%a.out
#SBATCH --error=HEADNET_%A_%a.err
#SBATCH --array=0-1799
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=10
#SBATCH --mem=5G

e=25

datasets=(cora_ml citeseer pubmed cora)
dims=(2 5 10 25 50)
seeds=({0..29})
exps=(lp_experiment recon_experiment rn_experiment)

num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}
num_exps=${#exps[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / (num_exps * num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / (num_exps * num_seeds) % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID / num_exps % num_seeds))
exp_id=$((SLURM_ARRAY_TASK_ID % num_exps))

dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}
exp=${exps[$exp_id]}

echo $dataset $dim $seed $exp

data_dir=datasets/${dataset}
if [ $exp == "lp_experiment" ]
then 
    edgelist=$(printf edgelists/${dataset}/seed=%03d/training_edges/edgelist.tsv ${seed})
elif [ $exp == "rn_experiment" ]
then 
    edgelist=$(printf nodes/${dataset}/seed=%03d/training_edges/edgelist.tsv ${seed})
else 
    edgelist=${data_dir}/edgelist.tsv.gz
fi
echo edgelist is $edgelist
features=${data_dir}/feats.csv.gz

embedding_dir=embeddings/${dataset}/$exp
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

        args=$(echo --edgelist ${edgelist} --features ${features} \
        --embedding ${embedding_dir} --seed ${seed} \
        --dim ${dim} --context-size 1 -e ${e})

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
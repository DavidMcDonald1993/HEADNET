#!/bin/bash

# experiments
for dataset in cora_ml citeseer pubmed
do
	for dim in 2 5 10 25 50
	do	
		for seed in {00..29}
		do
			for exp in recon_experiment lp_experiment rn_experiment
			do
				embedding_dir=$(printf \
				"embeddings/${dataset}/${exp}/seed=%03d/dim=%03d" ${seed} ${dim})

				if [ -f ${embedding_dir}/final_embedding.csv ] 
				then
					if [ ! -f ${embedding_dir}/final_embedding.csv.gz ]
					then 
						gzip ${embedding_dir}/final_embedding.csv
					fi
				else
					echo no embedding at ${embedding_dir}/final_embedding.csv
				fi

				if [ -f ${embedding_dir}/final_variance.csv ]
				then 
					if [ ! -f ${embedding_dir}/final_variance.csv.gz ]
					then 
						gzip ${embedding_dir}/final_variance.csv
					fi
				else
					echo no variance at ${embedding_dir}/final_variance.csv
				fi
			done
		done
	done
done
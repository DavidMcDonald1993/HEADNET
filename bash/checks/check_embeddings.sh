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

				for matrix in embedding variance
				do

					if [ -f ${embedding_dir}/final_${matrix}.csv.gz ] 
					then
						continue
					fi

					if [ -f ${embedding_dir}/final_${matrix}.csv ] 
					then
						gzip ${embedding_dir}/final_${matrix}.csv
					else
						echo no embedding at ${embedding_dir}/final_${matrix}.csv
					fi

				done
			done
		done
	done
done
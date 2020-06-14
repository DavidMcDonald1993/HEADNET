#!/bin/bash

for dataset in cora_ml citeseer pubmed cora 
do
	for dim in 5 10 25 50
	do	
		for seed in {0..29}
		do
			for exp in rn_experiment
			do
				for feat in feats
				do
					embedding_dir=$(printf \
					"embeddings/${dataset}/${feat}/${exp}/seed=%03d/dim=%03d" ${seed} ${dim})

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
done
#!/bin/bash

# datasets=({00..29})
datasets=({cora_ml,citeseer,pubmed})
seeds=({0..29})

for dataset in ${datasets[@]};
do
	output=nodes/${dataset}
	for seed in ${seeds[@]};
	do
		edgelist_f=$(printf "${output}/seed=%03d/training_edges/edgelist.tsv" ${seed} )

		if [ ! -f $edgelist_f  ]
		then
			echo $edgelist_f is missing 
		fi
	done
done
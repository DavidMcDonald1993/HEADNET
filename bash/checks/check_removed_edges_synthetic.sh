#!/bin/bash

datasets=({00..29})
seeds=({0..29})

for dataset in ${datasets[@]};
do

	output=edgelists/synthetic_scale_free/${dataset}

	for seed in ${seeds[@]};
	do
		edgelist_f=$(printf "${output}/seed=%03d/training_edges/graph.npz" ${seed} )

		if [ ! -f $edgelist_f  ]
		then
			echo $edgelist_f is missing 
		fi

	done
done
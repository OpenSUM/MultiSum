#!/bin/bash

set -v

ckpt=checkpoint
i=5
total=30
ckpt_dir=./experiments/${ckpt}
mkdir -p ${ckpt_dir}/summary
while [[ $i -lt $total ]]
do
	fp="${ckpt_dir}/best-${i}.data-00000-of-00001"
	while [[ ! -f ${fp} ]]
	do
		echo "File doesn't exist: ${fp}"
		sleep 5s
	done
	echo Decoding using checkpoint: $fp
	 python -u run.py --mode=test --init_checkpoint=${ckpt} --checkpoint_file=best-$i --num_gpus=4& 
	 sleep 5s
	 python -u run.py --mode=eval --init_checkpoint=${ckpt} --checkpoint_file=best-$i --num_gpus=4&
	 sleep 5s
	i=$[i+1]
done


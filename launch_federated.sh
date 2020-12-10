#!/usr/bin/env bash

function runexp {

gpu=${1}
model=${2}
lr=${3}
iters=${4}
latent=${5}
layers=${6}
dff=${7}

LD_LIBRARY_PATH="/opt/common/cudnn/cudnn-10.1-7.6.5/lib64:/opt/common/cuda/cuda-10.1.243/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

module unload cuda/10.2.89
module load cuda/10.1.243

expname=nwp-federated-${model}-lr_${lr}-iters_${iters}-layers_${layers}-latent_${latent}-dff_${dff}

cmd="
CUDA_VISIBLE_DEVICES=${gpu} python federated_trainer.py
 --task=stackoverflow_nwp --total_rounds=2
 --client_optimizer=adam --client_learning_rate=1e-3
    --client_batch_size=16  --server_optimizer=adam
 --server_learning_rate=1e-2 --clients_per_round=10
 --client_epochs_per_round=1 --experiment_name=nwp_federated

"

# --lr ${lr} --iters ${iters} --latent_size ${latent}
#    --layers ${layers}  --dff ${dff} --model ${model}
#  > logs/${expname}.log 2>&1 &

eval ${cmd}

echo logs/${expname}.log


}

# runexp  gpu model         lr      iters    latent  layers  dff
#runexp   0    lstm         2e-3      250796   670      2       256
#runexp   0    lstm         3e-3      250796   670      2       256
#runexp   1    lstm         4e-3      250796   670      2       256

#runexp   1    transformer  1e-3      250796     96   4       1536
#runexp   0    transformer  1e-3      250796     96   4       2048
runexp   0    transformer  1e-3      250796     96   3       1536
runexp   1    transformer  1e-3      250796     96   5       2048
runexp   2    transformer  1e-3      250796     96   3       1536
runexp   3    transformer  1e-3      250796     96   5       2048

#runexp   0    3e-4      250796    256   1

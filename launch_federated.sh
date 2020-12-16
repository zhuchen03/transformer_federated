#!/usr/bin/env bash

function runexp {

gpu=${1}
model=${2}
slr=${3}
clr=${4}
clients=${5}
latent=${6}
layers=${7}
dff=${8}

bsize=16

LD_LIBRARY_PATH="/opt/common/cudnn/cudnn-10.1-7.6.5/lib64:/opt/common/cuda/cuda-10.1.243/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

module unload cuda/10.2.89
module load cuda/10.1.243

expname=nwp-federated-${model}-slr_${slr}-clr_${clr}-clients_${clients}-layers_${layers}-latent_${latent}-dff_${dff}-bsize${bsize}

# --task=stackoverflow_nwp --total_rounds=1500  --client_batch_size=16
# --so_nwp_max_elements_per_user 1000  --clients_per_round=50
cmd="CUDA_VISIBLE_DEVICES=${gpu}
 python federated_trainer.py
  --task=stackoverflow_nwp --total_rounds=3000  --client_batch_size=${bsize}
  --so_nwp_model_type=${model}
 --so_nwp_max_elements_per_user 2000  --clients_per_round=${clients}
 --client_optimizer=adam --client_learning_rate=${clr}
 --client_lr_schedule=inv_sqrt_decay  --client_lr_warmup_steps=300 --client_lr_decay_steps=300 --client_lr_decay_rate=1
 --server_optimizer=adam --server_learning_rate=${slr}
 --server_lr_schedule=inv_sqrt_decay  --server_lr_warmup_steps=300 --server_lr_decay_steps=300 --server_lr_decay_rate=1
 --so_nwp_num_layers=${layers} --so_nwp_latent_size=${latent} --so_nwp_dff=${dff}
 --client_epochs_per_round=1 --experiment_name=nwp_federated
 --root_output_dir chks/${expname}
 > logs/${expname}.log 2>&1 &
"

# --lr ${lr} --iters ${iters} --latent_size ${latent}
#    --layers ${layers}  --dff ${dff} --model ${model}
#

eval ${cmd}

echo logs/${expname}.log


}

# runexp  gpu model          slr  clr   clients    latent  layers  dff
runexp   0    transformer   1e-3  4e-5   25         96      3       1536



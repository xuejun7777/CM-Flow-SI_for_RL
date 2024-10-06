#!/bin/bash

declare -a envs=(
                'halfcheetah-medium-v2'
                # 'hopper-medium-v2'
                # 'walker2d-medium-v2'
                # 'halfcheetah-medium-replay-v2'
                # 'hopper-medium-replay-v2'
                # 'walker2d-medium-replay-v2'
                # 'halfcheetah-medium-expert-v2'
                # 'hopper-medium-expert-v2'
                # 'walker2d-medium-expert-v2'
                # 'antmaze-umaze-v0'
                # 'antmaze-umaze-diverse-v0'
                # 'antmaze-medium-play-v0'
                # 'antmaze-medium-diverse-v0'
                # 'antmaze-large-play-v0'
                # 'antmaze-large-diverse-v0'
                # 'pen-human-v1'
                # 'pen-cloned-v1'
                # 'kitchen-complete-v0'
                # 'kitchen-partial-v0'
                # 'kitchen-mixed-v0'
)


declare -a methods=('offline')  # offline or online model selection
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
mkdir -p log/$DATE

# offline RL
declare -a seeds=($(seq 1 1 1))  # run 5 seeds for each env

for i in ${!envs[@]}; do
    for j in ${!methods[@]}; do
        for k in ${!seeds[@]}; do
            echo python offline.py --env_name ${envs[$i]}  --model stochastic_interpolants --exp si${seeds[$k]} --seed ${seeds[$k]} --ms ${methods[$j]} --save_best_model --early_stop --lr_decay output log to: log/$DATE/${envs[$i]}_${methods[$j]}_si_seed${seeds[$k]}.log &
            nohup python -W ignore  offline.py --env_name ${envs[$i]}  --model stochastic_interpolants --exp si${seeds[$k]} --seed ${seeds[$k]} --ms ${methods[$j]} --save_best_model --lr_decay --early_stop >> log/$DATE/${envs[$i]}_${methods[$j]}_si_seed${seeds[$k]}.log &        
            echo python offline.py --env_name ${envs[$i]}  --model consistency --exp consistency${seeds[$k]} --seed ${seeds[$k]} --ms ${methods[$j]} --save_best_model --lr_decay --early_stop output log to: log/$DATE/${envs[$i]}_${methods[$j]}_consis_seed${seeds[$k]}.log &
            nohup python -W ignore  offline.py --env_name ${envs[$i]}  --model consistency --exp consistency${seeds[$k]} --seed ${seeds[$k]} --ms ${methods[$j]} --save_best_model --lr_decay --early_stop >> log/$DATE/${envs[$i]}_${methods[$j]}_consis_seed${seeds[$k]}.log &        
            echo python offline.py --env_name ${envs[$i]}  --model diffusion --exp diffusion${seeds[$k]} --seed ${seeds[$k]} --ms ${methods[$j]} --save_best_model --lr_decay --early_stop output log to: log/$DATE/${envs[$i]}_${methods[$j]}_diff_seed${seeds[$k]}.log &
            nohup python -W ignore  offline.py --env_name ${envs[$i]}  --model diffusion --exp diffusion${seeds[$k]} --seed ${seeds[$k]} --ms ${methods[$j]} --save_best_model --lr_decay --early_stop >> log/$DATE/${envs[$i]}_${methods[$j]}_diff_seed${seeds[$k]}.log &        
        done
    done
done
wait
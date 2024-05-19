# birds_classification

## Install

1. conda create -n birds python=3.10
1. conda activate birds
1. python3 -m pip install -r requirements.txt
1. Install apex as [doc](https://github.com/NVIDIA/NeMo?tab=readme-ov-file#apex)

## Training process
1. train.sh (may be resume.sh if the machine crash)
1. finetune.sh (train with small categories)
1. eval.sh (get 'incorrect' samples for the model)
1. train_second.sh (train with 'incorrect' samples enhanced)

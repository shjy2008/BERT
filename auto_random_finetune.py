import random
import subprocess
import os
import shutil

PREF='sst'
possible_params = {
    "lr": [5e-6, 1e-5, 2e-5, 5e-5],
    "epochs": [10],
    "batch_size": [4, 8, 16, 32],
    "hidden_dropout_prob": [0.1, 0.2, 0.3, 0.5],
    "weight_decay": [0, 0.1, 0.01, 0.001],
    "POS_tag_enabled": [0],
    "dep_tag_enabled": [0],
    "use_MSE_loss": [0],
    "use_CORAL_loss": [1],
    "use_scheduler": [0, 1],
    "freeze_layers": [0],
    "load_existing_model": [1],
    "do_training": [1],
    "seed": [1234],
}

def get_random_params():
    return {key: random.choice(values) for key, values in possible_params.items()}

def build_command(finetune_model_path):
    params = get_random_params()

    command = f"""
python classifier.py \
    --use_gpu \
    --option finetune \
    --lr {params['lr']} \
    --epochs {params['epochs']} \
    --batch_size {params['batch_size']} \
    --hidden_dropout_prob {params['hidden_dropout_prob']} \
    --weight_decay {params['weight_decay']} \
    --POS_tag_enabled {params['POS_tag_enabled']} \
    --dep_tag_enabled {params['dep_tag_enabled']} \
    --use_MSE_loss {params['use_MSE_loss']} \
    --use_CORAL_loss {params['use_CORAL_loss']} \
    --use_scheduler {params['use_scheduler']} \
    --freeze_layers {params['freeze_layers']} \
    --load_existing_model {params['load_existing_model']} \
    --do_training {params['do_training']} \
    --seed {params['seed']} \
    --train "data/{PREF}-train.txt" \
    --dev "data/{PREF}-dev.txt" \
    --test "data/{PREF}-test.txt" \
    --dev_out "{PREF}-dev-output.txt" \
    --test_out "{PREF}-test-output.txt" \
    --filepath {finetune_model_path} | tee {PREF}-train-log.txt
    """

    return command

for i in range(10):
    print (f"Round: {i}")

    base_model_path = f"{PREF}-finetune-model-708k-epoch1.pt"
    finetune_model_path = f"random/{PREF}-finetune-model-{i}.pt"
    print (f"Copying from {base_model_path} to {finetune_model_path}")

    os.makedirs(os.path.dirname(finetune_model_path), exist_ok = True)
    shutil.copy(base_model_path, finetune_model_path)


    command = build_command(finetune_model_path)
    subprocess.call(command, shell = True)




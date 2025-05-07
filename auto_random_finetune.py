import random
import subprocess
import os
import shutil
import json

PREF='sst'
RANDOM_TRIES = 50
possible_params = {
    "lr": [5e-6, 1e-5, 2e-5, 5e-5],
    "epochs": [10],
    "batch_size": [4, 8, 16, 32],
    "hidden_dropout_prob": [0.1, 0.2, 0.3, 0.5],
    "weight_decay": [0, 0.1, 0.01, 0.001],
    "POS_tag_enabled": [0, 1],
    "dep_tag_enabled": [0, 1],
    "use_MSE_loss": [0],
    "use_CORAL_loss": [1],
    "use_scheduler": [0, 1],
    "freeze_layers": [0, 6, 9],
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


if __name__ == "__main__":
    best_dev_acc = 0
    best_test_acc = 0
    best_round_dev = 0
    best_round_test = 0

    for i in range(RANDOM_TRIES):
        print (f"Round: {i}", flush=True)

        base_model_path = f"{PREF}-finetune-model-708k-epoch1.pt"
        finetune_model_path = f"random/{PREF}-finetune-model-{i}.pt"
        print (f"Copying from {base_model_path} to {finetune_model_path}", flush=True)

        os.makedirs(os.path.dirname(finetune_model_path), exist_ok = True)
        # shutil.copy(base_model_path, finetune_model_path)


        command = build_command(finetune_model_path)
        output = subprocess.check_output(command, shell = True, text = True)
        results = json.loads(output)

        dev_acc = results["dev_acc"]
        test_acc = results["test_acc"]

        # Get the best round
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_round_dev = i
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_round_test = i
        
        print (f"The best dev_acc is {best_dev_acc} in round {best_round_dev}", flush=True)
        print (f"The best test_acc is {best_test_acc} in round {best_round_test}", flush=True)






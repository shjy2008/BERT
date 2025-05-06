#!/bin/bash
#SBATCH --job-name=DL2
#SBATCH --account=sheju347

# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=20

# #SBATCH --out=log.txt

#SBATCH --partition=aoraki_gpu
# #SBATCH --partition=aoraki_gpu_H100
# #SBATCH --partition=aoraki_gpu_A100_80GB
#SBATCH --gpus-per-node=1
#SBATCH --mem=60GB
#SBATCH --time=02:00:00

# echo "hello world"

# usual bash commands go below here:
echo "my script will now start"
# nvidia-smi
# sleep 10 # pretend to do something

conda init bash
source ~/.bashrc
conda --version

echo "conda init"

conda activate LLM

echo "conda acticate LLM"


PREF='cfimdb'
python classifier.py \
    --use_gpu \
    --option pretrain \
    --lr 1e-3 \
    --epochs 10 \
    --batch_size 8 \
    --hidden_dropout_prob 0.3 \
    --weight_decay 0 \
    --POS_tag_enabled 0 \
    --dep_tag_enabled 1 \
    --use_MSE_loss 0 \
    --use_CORAL_loss 0 \
    --use_shceduler 0 \
    --load_existing_model 1 \
    --do_training 0 \
    --seed 1234 \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_out "${PREF}-dev-output.txt" \
    --test_out "${PREF}-test-output.txt" \
    --filepath "${PREF}-pretrain-model.pt" | tee ${PREF}-train-log.txt
    

echo "my script has finished."

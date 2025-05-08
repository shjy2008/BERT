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


PREF='sst'
python classifier.py \
    --use_gpu \
    --batch_size 8 \
    --load_existing_model 1 \
    --do_training 0 \
    --seed 1234 \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_out "${PREF}-dev-output.txt" \
    --test_out "${PREF}-test-output.txt" \
    --filepath "random/saved/sst-finetune-model-563-581.pt" | tee ${PREF}-train-log.txt
    

echo "my script has finished."

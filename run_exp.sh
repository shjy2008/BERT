# Step 1. (Optional) Any preprocessing step, e.g., downloading pre-trained word embeddings


# Step 2. Train models on two datasets.
##  2.1. Run experiments on SST
PREF='sst'
python classifier.py \
    --use_gpu \
    --option finetune \
    --lr 1e-5 \
    --seed 1234 \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_out "${PREF}-dev-output.txt" \
    --test_out "${PREF}-test-output.txt" \
    --filepath "${PREF}-model.pt" | tee ${PREF}-train-log.txt

##  2.2 Run experiments on CF-IMDB
PREF='cfimdb'
python classifier.py \
    --use_gpu \
    --option finetune \
    --lr 1e-5 \
    --seed 1234 \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_out "${PREF}-dev-output.txt" \
    --test_out "${PREF}-test-output.txt" \
    --filepath "${PREF}-model.pt" | tee ${PREF}-train-log.txt


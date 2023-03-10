python -u main.py \
        --patience 15 \
        --train-path "/fsc/fsc_train.csv" \
        --valid-path "/fsc/fsc_valid.csv" \
        --test-path "/fsc/fsc_test.csv" \
        --audio-path "/fsc/fluent_speech_commands_dataset/" \
        --logging-file "logs/fsc.log" \
        --dict-path "/saved_models/libri960_64b_steps_300000.pt" \
        --slu-data "fsc" \
        --nclasses 31 \
        --batch-size 32 \
        --lr 0.00002 \
        --norm-epoch 3 \
        --pyr-layer 3 \
        --nlayer 6 \
        --nhead 12 \
        --sample-rate 8000 \
        --nspeech-feat 80 \
        --specaug \
        --cuda \
        --seed 1111

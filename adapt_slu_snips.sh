for i in 1..10 #folds
do
    python -u main.py \
            --patience 10 \
            --train-path "/snips/folds/train_${i}.csv" \
            --valid-path "/snips/folds/valid_${i}.csv" \
            --test-path "/snips/folds/test_${i}.csv" \
            --audio-path "/snips/" \
            --logging-file "logs/snips_slu_close.log" \
            --dict-path "/saved_models/libri960_64b_steps_300000.pt" \
            --slu-data "snips" \
            --batch-size 16 \
            --lr 0.00002 \
            --norm-epoch 3 \
            --pyr-layer 3 \
            --nlayer 6 \
            --nhead 12 \
            --sample-rate 8000 \
            --nspeech-feat 80 \
            --cuda \
            --seed 1111
done

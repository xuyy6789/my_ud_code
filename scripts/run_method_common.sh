
    for myweight_learning_rate in 0.06
    do
        for template_id in 0
        do
            for myseed in 144 
            do
                python -u src/run.py \
                --task train_and_save_embedding_visualization \
                --seed $myseed \
                --dataset "agnews" \
                --template_id $template_id \
                --num_train_samples_per_label 5 \
                --num_label_types 2 \
                --num_original_label_types 4 \
                --model_type roberta \
                --model_name_or_path "roberta-large" \
                --max_seq_length 128 \
                --criterion_type rank \
                --dic_dataset "datasets_round1" \
                --label_words_kb "label_words_common.txt" \
                --template_file "template.txt" \
                --round_num 0 \
                --file_train_dataset "train_2class_1_world.csv" \
                --file_valid_dataset "-" \
                --file_train_labels_dataset "-" \
                --file_test_dataset "-" \
                --do_valid 0 \
                --epoch 9 \
                --warmup_steps 0 \
                --learning_rate 2e-5 
            done
        done
    done

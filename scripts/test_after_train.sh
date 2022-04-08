
    for myweight_learning_rate in 0.06
    do
        for template_id in 0 
        do
            for myseed in 144 
            do
                python -u src/test_after_train.py \
                --task train_and_save_embedding_visualization \
                --seed $myseed \
                --dataset "dbpedia" \
                --template_id $template_id \
                --num_train_samples_per_label -1 \
                --num_label_types 2 \
                --num_original_label_types 4 \
                --model_type roberta \
                --model_name_or_path "roberta-large" \
                --max_seq_length 128 \
                --criterion_type rank \
                --dic_dataset "datasets_round1" \
                --label_words_kb "label_words_common.txt" \
                --template_file "template.txt" \
                --round_num 2 \
                --file_train_dataset "-" \
                --file_valid_dataset  "-" \
                --file_test_dataset "test.txt" \
                --file_test_labels_dataset "test_labels_2class_11_plant.txt" \
                --do_valid 0 \
                --epoch 0 \
                --best_k_ratio 0.5 \
                --warmup_steps 0 \
                --learning_rate 0
            done
        done
    done
# done

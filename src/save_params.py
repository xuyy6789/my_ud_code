import os
import torch
import pandas as pd
import numpy as np

def save_parameters(model, criterion_params, criterion, tokenizer=None):
    output_dir = "./model_save/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, 'module') else model

    model_to_save.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)
    torch.save(criterion.state_dict(), output_dir + "criterion.ckpt")

    torch.save(criterion_params, os.path.join(output_dir, "criterion_params.pt")) #将字符串写入文件中

    print("save finished!")

def save_train_data(selected_text_a, selected_text_b, selected_labels):
    # print(selected_labels)
    str_a = ''
    str_b = ''
    str_l = ''
    selected_text_a = [str_a.join(['"', i, '"']) for i in selected_text_a]
    selected_text_b = [str_b.join(['"', i, '"']) for i in selected_text_b]
    selected_labels = [str_l.join(['"', str(i+1), '"']) for i in selected_labels]
    list = []
    for i, item in enumerate(selected_text_a):
        list_i = []
        list_i.append(selected_labels[i])
        list_i.append(',')
        list_i.append(selected_text_a[i])
        list_i.append(',')
        list_i.append(selected_text_b[i])
        # print(list_i)
        str_i = ''
        str_i = str_i.join(list_i)
        list.append(str_i)
    np.savetxt("train.csv", list, delimiter=",", newline="\n", fmt ='%s')

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import xlwt

# from keras.preprocessing.sequence import pad_sequences

from configs import get_model_classes
from save_params import save_train_data

class AgnewsDataset():
    def __init__(self):
        pass

    def get_trainset_from_csv(self, args):
        df = pd.read_csv(os.path.join(args.dic_dataset, args.dataset, args.file_train_dataset), delimiter=',', 
                    header=None, names=['label', 'headline', 'body'])
        text_a = df.headline.values # type is ndarray
        text_b = df.body.values
        labels = df.label.values

        for i, text in enumerate(text_b):
            text_b[i] = text.replace('\\', ' ')
        for i, label in enumerate(labels):
            labels[i] = int(label) - 1


        # 根据label分类，并select samples（小样本，所以取num_train_samples_per_label个）
        selected_ids = []
        ids_per_label = [[] for x in range(max(labels) + 1)] # [[label0], [label1], [label2], [label3]]
        print("ids_per_label", ids_per_label)
        for idx, label in enumerate(labels):
            ids_per_label[label].append(idx) # 每个label，记录下这句话对应的编号
        for ids in ids_per_label: # 一行一行
            tmp = np.array(ids)
            np.random.shuffle(tmp) # 打乱顺序
            selected_ids.extend(tmp[:args.num_train_samples_per_label].tolist()) # 每个label取args.num_examples_per_label个例子——变成了一维的
        selected_ids = np.array(selected_ids)
        np.random.shuffle(selected_ids) # 打乱前：每个label在一堆；打乱后：label混合
        args.selected_ids = selected_ids # 一个list = [2, 1, 3, 1, 0, 3, 1, ...]

        selected_text_a = [text_a[idx] for idx in selected_ids]
        selected_text_b = [text_b[idx] for idx in selected_ids]
        selected_labels = [labels[idx] for idx in selected_ids]

        # 保存train dataset
        save_train_data(selected_text_a, selected_text_b, selected_labels)
        
        print(selected_labels)
        print("================train selected_ids")
        print(selected_ids)

        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]

        print(model_config)
        print(type(args.model_name_or_path))

        tokenizer = model_config['tokenizer'].from_pretrained(
            args.model_name_or_path
        )
        self.tokenizer = tokenizer
        
        print(tokenizer.mask_token) # test
        
        composed = []
        num_all = args.num_label_types * args.num_train_samples_per_label
        print("num_all: ", num_all)
        print(len(selected_text_a))
        for i in range(num_all):
            str = ""
            for x in args.template_text:
                if x == "<mask>":
                    str += tokenizer.mask_token
                    str += " "
                elif x == "<Mask>":
                    str += tokenizer.mask_token
                    str += " "
                elif x == "<a>":
                    str += selected_text_a[i]
                    str += " "
                elif x == "<b>":
                    str += selected_text_b[i]
                    str += " "
                else:
                    str += x
                    str += " "
            composed.append(str)
        
        # print(composed[2])

        composed = np.array(composed)

        print(type(composed))
        # print(composed[2])

        composed_encode = [tokenizer.encode(sent, add_special_tokens=False) for sent in composed]

        print(type(composed_encode))
        print(composed_encode[1:3])
        # print(composed_encode[2])



        # truncate---裁剪 + 加入特殊tokens + 填充 + attention_mask层等
        token_type_ids = []
        attention_mask = []
        mlm_labels = []
        for i, list in enumerate(composed_encode):
            num_to_remove = len(list) - args.max_seq_length + 2 # 给特殊符号留位置
            while num_to_remove > 0:
                list.pop(-1)
                num_to_remove -= 1
            # 加特殊tokens + 填充
            composed_encode[i] = tokenizer.build_inputs_with_special_tokens(list)
            att_mask = [1] * len(composed_encode[i])
            token_type_ids_temp = tokenizer.create_token_type_ids_from_sequences(composed_encode[i][2:])
            padding_length = args.max_seq_length - len(composed_encode[i])
            composed_encode[i] = composed_encode[i] + ([tokenizer.pad_token_id] * padding_length)
            # get 其他需要的层
            att_mask = att_mask + ([0] * padding_length)
            # print(len(att_mask))
            attention_mask.append(att_mask)

            # 特别的，create_token_type_ids_from_sequences 的 return len(cls + token_ids_0 + sep) * [0]，
            # 所以要在添加cls和sep前做
            
            token_type_ids_temp = token_type_ids_temp + ([0] * padding_length)
            # print(len(token_type_ids_temp))
            token_type_ids.append(token_type_ids_temp)

            flags = [-1] * len(composed_encode[i])
            flags_idx = composed_encode[i].index(tokenizer.mask_token_id) # 找mask的位置
            flags[flags_idx] = 1 # 特定地方为1
            mlm_labels.append(flags) # 全部都是-1，mask的地方为1

        input_ids = composed_encode
        labels = selected_labels

        # 转tensor
        input_ids = torch.tensor(input_ids).long()
        token_type_ids = torch.tensor(token_type_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        mlm_labels = torch.tensor(mlm_labels).long()
        labels = torch.tensor(labels).long()

        # 将label存成文件
        # book = xlwt.Workbook()
        # sheet = book.add_sheet(u'sheet1',cell_overwrite_ok=True)
        # sheet.write(0, 0, 'test_data')
        # sheet.write(0, 1, 'label')

        # for i, num in enumerate(labels):
        #     num = num.item()
        #     sheet.write(i + 1, 0, i + 1)
        #     sheet.write(i + 1, 1, num)

        # book.save('./result/labels.xls')

        # 使用dataloader
        train_dataset = TensorDataset(input_ids, token_type_ids, attention_mask, labels, mlm_labels)
        train_batch_size = 16
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
        args.train_dataloader = train_dataloader

        print(args.train_dataloader)


    def get_validset_from_csv(self, args, exclude):
        df = pd.read_csv(os.path.join(args.dic_dataset, args.dataset, args.file_train_dataset), delimiter=',', 
                    header=None, names=['label', 'headline', 'body'])
        text_a = df.headline.values # type is ndarray
        text_b = df.body.values
        labels = df.label.values

        for i, text in enumerate(text_b):
            text_b[i] = text.replace('\\', ' ')
        for i, label in enumerate(labels):
            labels[i] = int(label) - 1


        # 根据label分类，并select samples（小样本，所以取num_train_samples_per_label个）
        
        selected_ids = []
        ids_per_label = [[] for x in range(max(labels) + 1)] # [[label0], [label1], [label2], [label3]]
        for idx, label in enumerate(labels):
            if idx not in exclude:
                ids_per_label[label].append(idx) # 每个label，记录下这句话对应的编号
        for ids in ids_per_label: # 一行一行
            tmp = np.array(ids)
            np.random.shuffle(tmp) # 打乱顺序
            selected_ids.extend(tmp[:args.num_train_samples_per_label].tolist()) # 每个label取args.num_examples_per_label个例子——变成了一维的
        selected_ids = np.array(selected_ids)
        np.random.shuffle(selected_ids) # 打乱前：每个label在一堆；打乱后：label混合
        args.selected_ids = selected_ids # 一个list = [2, 1, 3, 1, 0, 3, 1, ...]

        selected_text_a = [text_a[idx] for idx in selected_ids]
        selected_text_b = [text_b[idx] for idx in selected_ids]
        selected_labels = [labels[idx] for idx in selected_ids]
        


        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]

        print(model_config)
        print(type(args.model_name_or_path))

        tokenizer = model_config['tokenizer'].from_pretrained(
            args.model_name_or_path
        )
        self.tokenizer = tokenizer
        
        print(tokenizer.mask_token) # test
        
        composed = []
        num_all = args.num_label_types * args.num_train_samples_per_label
        print("selected_text_a: ", len(selected_text_a))
        for i in range(num_all):
            str = ""
            for x in args.template_text:
                if x == "<mask>":
                    str += tokenizer.mask_token
                    str += " "
                elif x == "<Mask>":
                    str += tokenizer.mask_token
                    str += " "
                elif x == "<a>":
                    str += selected_text_a[i]
                    str += " "
                elif x == "<b>":
                    str += selected_text_b[i]
                    str += " "
                else:
                    str += x
                    str += " "
            composed.append(str)
        
        print(composed[2])

        composed = np.array(composed)


        composed_encode = [tokenizer.encode(sent, add_special_tokens=False) for sent in composed]




        # truncate---裁剪 + 加入特殊tokens + 填充 + attention_mask层等
        token_type_ids = []
        attention_mask = []
        mlm_labels = []
        for i, list in enumerate(composed_encode):
            num_to_remove = len(list) - args.max_seq_length + 2 # 给特殊符号留位置
            while num_to_remove > 0:
                list.pop(-1)
                num_to_remove -= 1
            # 加特殊tokens + 填充
            composed_encode[i] = tokenizer.build_inputs_with_special_tokens(list)
            att_mask = [1] * len(composed_encode[i])
            token_type_ids_temp = tokenizer.create_token_type_ids_from_sequences(composed_encode[i][2:])
            padding_length = args.max_seq_length - len(composed_encode[i])
            composed_encode[i] = composed_encode[i] + ([tokenizer.pad_token_id] * padding_length)
            # get 其他需要的层
            att_mask = att_mask + ([0] * padding_length)
            # print(len(att_mask))
            attention_mask.append(att_mask)
            # attention_mask.append([1] * len(composed_encode[i]))
            # token_type_ids.append(tokenizer.create_token_type_ids_from_sequences(composed_encode[i]))
            # 特别的，create_token_type_ids_from_sequences 的 return len(cls + token_ids_0 + sep) * [0]，
            # 所以要在添加cls和sep前做
            
            token_type_ids_temp = token_type_ids_temp + ([0] * padding_length)
            token_type_ids.append(token_type_ids_temp)

            flags = [-1] * len(composed_encode[i])
            flags_idx = composed_encode[i].index(tokenizer.mask_token_id) # 找mask的位置
            flags[flags_idx] = 1 # 特定地方为1
            mlm_labels.append(flags) # 全部都是-1，mask的地方为1

        input_ids = composed_encode
        labels = selected_labels

        # 转tensor
        input_ids = torch.tensor(input_ids).long()
        token_type_ids = torch.tensor(token_type_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        mlm_labels = torch.tensor(mlm_labels).long()
        labels = torch.tensor(labels).long()


        # 将label存成文件
        # book = xlwt.Workbook()
        # sheet = book.add_sheet(u'sheet1',cell_overwrite_ok=True)
        # sheet.write(0, 0, 'test_data')
        # sheet.write(0, 1, 'label')

        # for i, num in enumerate(labels):
        #     num = num.item()
        #     sheet.write(i + 1, 0, i + 1)
        #     sheet.write(i + 1, 1, num)

        # book.save('./result/labels.xls')

        # 使用dataloader
        valid_dataset = TensorDataset(input_ids, token_type_ids, attention_mask, labels, mlm_labels)
        valid_batch_size = 16
        valid_sampler = RandomSampler(valid_dataset)
        valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=valid_batch_size)
        args.valid_dataloader = valid_dataloader

        print(args.valid_dataloader)



    def get_template(args):
        temps = {}
        template_file = open(os.path.join(args.dic_dataset, args.dataset, args.template_file),'r') 
        templates = [line.strip().split() for line in template_file]
        template_id = args.template_id
        template = templates[template_id]
        args.template_text = template
        print(args.template_text)
    

    def get_label_words(self, args): # 处理label对应的label words
        """get all label words and encoding, then save self.encode_all_label_words"""
        all_label_words = {} # 字典：每个label一个list
        with open(os.path.join(args.dic_dataset, args.dataset, args.label_words_kb),'r') as f:
            myline = f.readlines() # 一行一组label words，即[label, label words(多个)]
            for idx, line in enumerate(myline):
                idx_label_words = line.strip().replace(",", " ").split()
                all_label_words[idx] = idx_label_words

        encode_all_label_words = []
        for i in range(len(all_label_words)):
            i_words = []
            for word in all_label_words[i]:

                word = self.tokenizer.encode(word, add_special_tokens=False) # 不加[CLS]和[SEP]——一个word变成[7506, 41009, 5183] 或者 [4656]
                i_words.append(word)
            encode_all_label_words.append(i_words)



        self.encode_all_label_words = encode_all_label_words
    
    
    def get_testset_from_csv(self, args):
        df = pd.read_csv(os.path.join(args.dic_dataset, args.dataset, args.file_test_dataset), delimiter=',', 
                    header=None, names=['label', 'headline', 'body'])
        text_a = df.headline.values # type is ndarray
        text_b = df.body.values
        labels = df.label.values
        
        for i, text in enumerate(text_b):
            text_b[i] = text.replace('\\', ' ')
        for i, label in enumerate(labels):
            labels[i] = int(label) - 1
       

        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]

        # print(model_config)
        # print(type(args.model_name_or_path))

        tokenizer = model_config['tokenizer'].from_pretrained(
            args.model_name_or_path
        )
        self.tokenizer = tokenizer
        
        # print(tokenizer.mask_token) # test
        
        composed = []
        num_all = len(text_a)
        for i in range(num_all):
            str = ""
            for x in args.template_text:
                if x == "<mask>":
                    str += tokenizer.mask_token
                    str += " "
                elif x == "<Mask>":
                    str += tokenizer.mask_token
                    str += " "
                elif x == "<a>":
                    str += text_a[i]
                    str += " "
                elif x == "<b>":
                    str += text_b[i]
                    str += " "
                else:
                    str += x
                    str += " "
            composed.append(str)
        
        # print(composed[2])

        composed = np.array(composed)

        # print(type(composed))
        # print(composed[2])

        composed_encode = [tokenizer.encode(sent, add_special_tokens=False) for sent in composed]

        # print(type(composed_encode))
        # print(composed_encode[1:3])

        # truncate---裁剪 + 加入特殊tokens + 填充 + attention_mask层等
        token_type_ids = []
        attention_mask = []
        mlm_labels = []
        for i, list in enumerate(composed_encode):
            num_to_remove = len(list) - args.max_seq_length + 2 # 给特殊符号留位置
            while num_to_remove > 0:
                list.pop(-1)
                num_to_remove -= 1
            # 加特殊tokens + 填充
            composed_encode[i] = tokenizer.build_inputs_with_special_tokens(list)
            att_mask = [1] * len(composed_encode[i])
            token_type_ids_temp = tokenizer.create_token_type_ids_from_sequences(composed_encode[i][2:])
            padding_length = args.max_seq_length - len(composed_encode[i])
            composed_encode[i] = composed_encode[i] + ([tokenizer.pad_token_id] * padding_length)
            # get 其他需要的层
            att_mask = att_mask + ([0] * padding_length)
            # print(len(att_mask))
            attention_mask.append(att_mask)

            
            token_type_ids_temp = token_type_ids_temp + ([0] * padding_length)
            token_type_ids.append(token_type_ids_temp)

            flags = [-1] * len(composed_encode[i])
            flags_idx = composed_encode[i].index(tokenizer.mask_token_id) # 找mask的位置
            flags[flags_idx] = 1 # 特定地方为1
            mlm_labels.append(flags) # 全部都是-1，mask的地方为1

        input_ids = composed_encode
        labels = labels

        # 转tensor
        input_ids = torch.tensor(input_ids).long()
        token_type_ids = torch.tensor(token_type_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        mlm_labels = torch.tensor(mlm_labels).long()
        labels = torch.tensor(labels).long()


        test_dataset = TensorDataset(input_ids, token_type_ids, attention_mask, labels, mlm_labels)
        test_batch_size = 16
        test_sampler = RandomSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size)
        args.test_dataloader = test_dataloader

        print(args.test_dataloader)
        print(len(labels))
        print(len(args.test_dataloader))


class DBPediaDataset():
    def __init__(self):
        pass

    def get_trainset_from_csv(self, args):
        # examples = []
        label_file  = open(os.path.join(args.dic_dataset, args.dataset, args.file_train_labels_dataset),'r') 
        labels  = [int(x.strip()) for x in label_file.readlines()]
        text_a = []
        text_b = []
        with open(os.path.join(args.dic_dataset, args.dataset, args.file_train_dataset),'r') as fin:
            for idx, line in enumerate(fin):
                splited = line.strip().split(". ")
                text_a_i, text_b_i = splited[0], splited[1:]
                text_a_i = text_a_i+"."
                text_b_i = ". ".join(text_b_i)
                text_a.append(text_a_i)
                text_b.append(text_b_i)
       

        # 根据label分类，并select samples（小样本，所以取num_train_samples_per_label个）
        selected_ids = []
        ids_per_label = [[] for x in range(max(labels) + 1)] # [[label0], [label1], [label2], [label3]]
        print("ids_per_label", ids_per_label)
        for idx, label in enumerate(labels):
            ids_per_label[label].append(idx) # 每个label，记录下这句话对应的编号
        for ids in ids_per_label: # 一行一行
            tmp = np.array(ids)
            np.random.shuffle(tmp) # 打乱顺序
            selected_ids.extend(tmp[:args.num_train_samples_per_label].tolist()) # 每个label取args.num_examples_per_label个例子——变成了一维的
        selected_ids = np.array(selected_ids)
        np.random.shuffle(selected_ids) # 打乱前：每个label在一堆；打乱后：label混合
        args.selected_ids = selected_ids # 一个list = [2, 1, 3, 1, 0, 3, 1, ...]

        selected_text_a = [text_a[idx] for idx in selected_ids]
        selected_text_b = [text_b[idx] for idx in selected_ids]
        selected_labels = [labels[idx] for idx in selected_ids]

        # 保存train dataset
        save_train_data(selected_text_a, selected_text_b, selected_labels)
        
        print(selected_labels)
        print("================train selected_ids")
        print(selected_ids)

        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]

        print(model_config)
        print(type(args.model_name_or_path))

        tokenizer = model_config['tokenizer'].from_pretrained(
            args.model_name_or_path
        )
        self.tokenizer = tokenizer
        
        print(tokenizer.mask_token) # test
        
        composed = []
        num_all = args.num_label_types * args.num_train_samples_per_label
        print("num_all: ", num_all)
        print(len(selected_text_a))
        for i in range(num_all):
            str = ""
            for x in args.template_text:
                if x == "<mask>":
                    str += tokenizer.mask_token
                    str += " "
                elif x == "<Mask>":
                    str += tokenizer.mask_token
                    str += " "
                elif x == "<a>":
                    str += selected_text_a[i]
                    str += " "
                elif x == "<b>":
                    str += selected_text_b[i]
                    str += " "
                else:
                    str += x
                    str += " "
            composed.append(str)
        
        print(composed[2])

        composed = np.array(composed)

        print(type(composed))
        # print(composed[2])

        composed_encode = [tokenizer.encode(sent, add_special_tokens=False) for sent in composed]

        print(type(composed_encode))
        print(composed_encode[1:3])


        # truncate---裁剪 + 加入特殊tokens + 填充 + attention_mask层等
        token_type_ids = []
        attention_mask = []
        mlm_labels = []
        for i, list in enumerate(composed_encode):
            num_to_remove = len(list) - args.max_seq_length + 2 # 给特殊符号留位置
            while num_to_remove > 0:
                list.pop(-1)
                num_to_remove -= 1
            # 加特殊tokens + 填充
            composed_encode[i] = tokenizer.build_inputs_with_special_tokens(list)
            att_mask = [1] * len(composed_encode[i])
            token_type_ids_temp = tokenizer.create_token_type_ids_from_sequences(composed_encode[i][2:])
            padding_length = args.max_seq_length - len(composed_encode[i])
            composed_encode[i] = composed_encode[i] + ([tokenizer.pad_token_id] * padding_length)
            # get 其他需要的层
            att_mask = att_mask + ([0] * padding_length)
            # print(len(att_mask))
            attention_mask.append(att_mask)

            
            token_type_ids_temp = token_type_ids_temp + ([0] * padding_length)
            # print(len(token_type_ids_temp))
            token_type_ids.append(token_type_ids_temp)

            flags = [-1] * len(composed_encode[i])
            flags_idx = composed_encode[i].index(tokenizer.mask_token_id) # 找mask的位置
            flags[flags_idx] = 1 # 特定地方为1
            mlm_labels.append(flags) # 全部都是-1，mask的地方为1

        input_ids = composed_encode
        labels = selected_labels

        # 转tensor
        input_ids = torch.tensor(input_ids).long()
        token_type_ids = torch.tensor(token_type_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        mlm_labels = torch.tensor(mlm_labels).long()
        labels = torch.tensor(labels).long()



        print(labels)
        print(type(labels))

        # 将label存成文件
        # book = xlwt.Workbook()
        # sheet = book.add_sheet(u'sheet1',cell_overwrite_ok=True)
        # sheet.write(0, 0, 'test_data')
        # sheet.write(0, 1, 'label')

        # for i, num in enumerate(labels):
        #     num = num.item()
        #     sheet.write(i + 1, 0, i + 1)
        #     sheet.write(i + 1, 1, num)

        # book.save('./result/labels.xls')

        # 使用dataloader
        train_dataset = TensorDataset(input_ids, token_type_ids, attention_mask, labels, mlm_labels)
        train_batch_size = 16
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
        args.train_dataloader = train_dataloader

        print(args.train_dataloader)

        


    def get_validset_from_csv(self, args, exclude):
        label_file  = open(os.path.join(args.dic_dataset, args.dataset, args.file_train_labels_dataset),'r') 
        labels  = [int(x.strip()) for x in label_file.readlines()]
        text_a = []
        text_b = []
        with open(os.path.join(args.dic_dataset, args.dataset, args.file_train_dataset),'r') as fin:
            for idx, line in enumerate(fin):
                splited = line.strip().split(". ")
                text_a_i, text_b_i = splited[0], splited[1:]
                text_a_i = text_a_i+"."
                text_b_i = ". ".join(text_b_i)
                text_a.append(text_a_i)
                text_b.append(text_b_i)


        # 根据label分类，并select samples（小样本，所以取num_train_samples_per_label个）
        
        selected_ids = []
        ids_per_label = [[] for x in range(max(labels) + 1)] # [[label0], [label1], [label2], [label3]]
        for idx, label in enumerate(labels):
            if idx not in exclude:
                ids_per_label[label].append(idx) # 每个label，记录下这句话对应的编号
        for ids in ids_per_label: # 一行一行
            tmp = np.array(ids)
            np.random.shuffle(tmp) # 打乱顺序
            selected_ids.extend(tmp[:args.num_train_samples_per_label].tolist()) # 每个label取args.num_examples_per_label个例子——变成了一维的
        selected_ids = np.array(selected_ids)
        np.random.shuffle(selected_ids) # 打乱前：每个label在一堆；打乱后：label混合
        args.selected_ids = selected_ids # 一个list = [2, 1, 3, 1, 0, 3, 1, ...]

        selected_text_a = [text_a[idx] for idx in selected_ids]
        selected_text_b = [text_b[idx] for idx in selected_ids]
        selected_labels = [labels[idx] for idx in selected_ids]
        
        print(selected_labels)
        print("================valid selected_ids")
        print(selected_ids)
        # f = open('./dataset_few_shot/valid_fourclass_144.csv', 'w')
        # for i in range(len(selected_labels)):
        #     f.write('"%d","%s","%s"\n' % (selected_labels[i], selected_text_a[i], selected_text_b[i]))
        #     # print("{}, {}, {}" % selected_labels, selected_text_a, selected_text_b)
        # f.close()

        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]

        print(model_config)
        print(type(args.model_name_or_path))

        tokenizer = model_config['tokenizer'].from_pretrained(
            args.model_name_or_path
        )
        self.tokenizer = tokenizer
        
        print(tokenizer.mask_token) # test
        
        composed = []
        num_all = args.num_label_types * args.num_train_samples_per_label
        print("selected_text_a: ", len(selected_text_a))
        for i in range(num_all):
            str = ""
            for x in args.template_text:
                if x == "<mask>":
                    str += tokenizer.mask_token
                    str += " "
                elif x == "<Mask>":
                    str += tokenizer.mask_token
                    str += " "
                elif x == "<a>":
                    str += selected_text_a[i]
                    str += " "
                elif x == "<b>":
                    str += selected_text_b[i]
                    str += " "
                else:
                    str += x
                    str += " "
            composed.append(str)
        
        print(composed[2])

        composed = np.array(composed)

        print(type(composed))
        # print(composed[2])

        composed_encode = [tokenizer.encode(sent, add_special_tokens=False) for sent in composed]



        # truncate---裁剪 + 加入特殊tokens + 填充 + attention_mask层等
        token_type_ids = []
        attention_mask = []
        mlm_labels = []
        for i, list in enumerate(composed_encode):
            num_to_remove = len(list) - args.max_seq_length + 2 # 给特殊符号留位置
            while num_to_remove > 0:
                list.pop(-1)
                num_to_remove -= 1
            # 加特殊tokens + 填充
            composed_encode[i] = tokenizer.build_inputs_with_special_tokens(list)
            att_mask = [1] * len(composed_encode[i])
            token_type_ids_temp = tokenizer.create_token_type_ids_from_sequences(composed_encode[i][2:])
            padding_length = args.max_seq_length - len(composed_encode[i])
            composed_encode[i] = composed_encode[i] + ([tokenizer.pad_token_id] * padding_length)
            # get 其他需要的层
            att_mask = att_mask + ([0] * padding_length)
            # print(len(att_mask))
            attention_mask.append(att_mask)
            # attention_mask.append([1] * len(composed_encode[i]))
            # token_type_ids.append(tokenizer.create_token_type_ids_from_sequences(composed_encode[i]))
            # 特别的，create_token_type_ids_from_sequences 的 return len(cls + token_ids_0 + sep) * [0]，
            # 所以要在添加cls和sep前做
            
            token_type_ids_temp = token_type_ids_temp + ([0] * padding_length)
            token_type_ids.append(token_type_ids_temp)

            flags = [-1] * len(composed_encode[i])
            flags_idx = composed_encode[i].index(tokenizer.mask_token_id) # 找mask的位置
            flags[flags_idx] = 1 # 特定地方为1
            mlm_labels.append(flags) # 全部都是-1，mask的地方为1

        input_ids = composed_encode
        labels = selected_labels

        # 转tensor
        input_ids = torch.tensor(input_ids).long()
        token_type_ids = torch.tensor(token_type_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        mlm_labels = torch.tensor(mlm_labels).long()
        labels = torch.tensor(labels).long()



        # 将label存成文件
        # book = xlwt.Workbook()
        # sheet = book.add_sheet(u'sheet1',cell_overwrite_ok=True)
        # sheet.write(0, 0, 'test_data')
        # sheet.write(0, 1, 'label')

        # for i, num in enumerate(labels):
        #     num = num.item()
        #     sheet.write(i + 1, 0, i + 1)
        #     sheet.write(i + 1, 1, num)

        # book.save('./result/labels.xls')

        # 使用dataloader
        valid_dataset = TensorDataset(input_ids, token_type_ids, attention_mask, labels, mlm_labels)
        valid_batch_size = 16
        valid_sampler = RandomSampler(valid_dataset)
        valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=valid_batch_size)
        args.valid_dataloader = valid_dataloader

        print(args.valid_dataloader)



    def get_template(args):
        temps = {}
        template_file = open(os.path.join(args.dic_dataset, args.dataset, args.template_file),'r') 
        templates = [line.strip().split() for line in template_file]
        template_id = args.template_id
        template = templates[template_id]
        args.template_text = template
        print(args.template_text)
    

    def get_label_words(self, args): # 处理label对应的label words
        """get all label words and encoding, then save self.encode_all_label_words"""
        all_label_words = {} # 字典：每个label一个list
        with open(os.path.join(args.dic_dataset, args.dataset, args.label_words_kb),'r') as f:
            myline = f.readlines() # 一行一组label words，即[label, label words(多个)]
            for idx, line in enumerate(myline):
                idx_label_words = line.strip().replace(",", " ").split()
                all_label_words[idx] = idx_label_words
        # 实际上已经手动删除
        # self.all_label_words = self.delete_common_words(self.label_id_2_name) # {0: ['great', 'excellent', 'good', 'good', 'good', 'fantastic']}
        encode_all_label_words = []
        for i in range(len(all_label_words)):
            i_words = []
            for word in all_label_words[i]:
                # if not self.temps["mask_first"]:
                #     word = " " + word ###########   一个trick
                word = self.tokenizer.encode(word, add_special_tokens=False) # 不加[CLS]和[SEP]——一个word变成[7506, 41009, 5183] 或者 [4656]
                i_words.append(word)
            encode_all_label_words.append(i_words)



        self.encode_all_label_words = encode_all_label_words
    
    
    def get_testset_from_csv(self, args):

        label_file  = open(os.path.join(args.dic_dataset, args.dataset, args.file_test_labels_dataset),'r') 
        labels  = [int(x.strip()) for x in label_file.readlines()]
        text_a = []
        text_b = []
        with open(os.path.join(args.dic_dataset, args.dataset, args.file_test_dataset),'r') as fin:
            for idx, line in enumerate(fin):
                splited = line.strip().split(". ")
                text_a_i, text_b_i = splited[0], splited[1:]
                text_a_i = text_a_i+"."
                text_b_i = ". ".join(text_b_i)
                text_a.append(text_a_i)
                text_b.append(text_b_i)
        

       

        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]

        # print(model_config)
        # print(type(args.model_name_or_path))

        tokenizer = model_config['tokenizer'].from_pretrained(
            args.model_name_or_path
        )
        self.tokenizer = tokenizer
        
        # print(tokenizer.mask_token) # test
        
        composed = []
        num_all = len(text_a)
        for i in range(num_all):
            str = ""
            for x in args.template_text:
                if x == "<mask>":
                    str += tokenizer.mask_token
                    str += " "
                elif x == "<Mask>":
                    str += tokenizer.mask_token
                    str += " "
                elif x == "<a>":
                    str += text_a[i]
                    str += " "
                elif x == "<b>":
                    str += text_b[i]
                    str += " "
                else:
                    str += x
                    str += " "
            composed.append(str)
        
        # print(composed[2])

        composed = np.array(composed)

        # print(type(composed))
        # print(composed[2])

        composed_encode = [tokenizer.encode(sent, add_special_tokens=False) for sent in composed]

        # print(type(composed_encode))
        # print(composed_encode[1:3])

        # truncate---裁剪 + 加入特殊tokens + 填充 + attention_mask层等
        token_type_ids = []
        attention_mask = []
        mlm_labels = []
        for i, list in enumerate(composed_encode):
            num_to_remove = len(list) - args.max_seq_length + 2 # 给特殊符号留位置
            while num_to_remove > 0:
                list.pop(-1)
                num_to_remove -= 1
            # 加特殊tokens + 填充
            composed_encode[i] = tokenizer.build_inputs_with_special_tokens(list)
            att_mask = [1] * len(composed_encode[i])
            token_type_ids_temp = tokenizer.create_token_type_ids_from_sequences(composed_encode[i][2:])
            padding_length = args.max_seq_length - len(composed_encode[i])
            composed_encode[i] = composed_encode[i] + ([tokenizer.pad_token_id] * padding_length)
            # get 其他需要的层
            att_mask = att_mask + ([0] * padding_length)
            # print(len(att_mask))
            attention_mask.append(att_mask)

            
            token_type_ids_temp = token_type_ids_temp + ([0] * padding_length)
            token_type_ids.append(token_type_ids_temp)

            flags = [-1] * len(composed_encode[i])
            flags_idx = composed_encode[i].index(tokenizer.mask_token_id) # 找mask的位置
            flags[flags_idx] = 1 # 特定地方为1
            mlm_labels.append(flags) # 全部都是-1，mask的地方为1

        input_ids = composed_encode
        labels = labels

        # 转tensor
        input_ids = torch.tensor(input_ids).long()
        token_type_ids = torch.tensor(token_type_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        mlm_labels = torch.tensor(mlm_labels).long()
        labels = torch.tensor(labels).long()


        test_dataset = TensorDataset(input_ids, token_type_ids, attention_mask, labels, mlm_labels)
        test_batch_size = 16
        test_sampler = RandomSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size)
        args.test_dataloader = test_dataloader


    

Datasets = {
    'agnews': AgnewsDataset,
    'dbpedia': DBPediaDataset
}

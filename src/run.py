import random
from re import A
from sched import scheduler
from secrets import token_bytes
import numpy as np
import torch
import torch.nn as nn
import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from configs import get_args_parser, get_model_classes
from datasets import Datasets
from criterion import get_criterion_classes
from save_params import save_parameters

args = get_args_parser()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

class Runner(object):
    def __init__(self, args):
        super(Runner, self).__init__()
        self.args = args
        set_seed(args.seed)
        # print(args.seed)
        # print(args.dataset)
        self.datasetclass = Datasets[args.dataset]
        self.datasetclass.get_template(args)
        self.datasetclass.get_trainset_from_csv(self, args) # 得到了args.train_dataloader
        if args.do_valid == 1:
            self.datasetclass.get_validset_from_csv(self, args, args.selected_ids)

        self.datasetclass.get_label_words(self, args) # 存入self.encode_all_label_words
        

    def train_model(self, args):
        model_class = get_model_classes()[args.model_type]
        if args.round_num == 0:
            model_path = args.model_name_or_path
        else:
            model_path = "model_save"
        model = model_class['model'].from_pretrained(
            # args.model_name_or_path
            model_path
        )

        # model_temp = model
        
        # model= nn.DataParallel(model)
        # model.to(device)

        self.model = model

        epochs = args.epoch #
        total_steps = len(args.train_dataloader) * epochs
        
        print(total_steps) # 12

        

        criterion_class = get_criterion_classes()[args.criterion_type]
        criterion = criterion_class(args=args, encode_all_label_words=self.encode_all_label_words)
        self.criterion = criterion
        
        # 01号weight训练方法
        optimizer = AdamW(
            [{"params": model.parameters()},
             {"params": criterion.parameters()}], 
            lr = args.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)


        # 训练过程
        save_logits = []
        # model.train()
        # criterion.train()
        for i in range(0, epochs):
        # with torch.no_grad():
            print("======== The epochs is {%d} ========" % (i+1))
            total_loss = 0
            for step, batch in enumerate(args.train_dataloader):
                print("======== The batch step is {%d} ========" % step)
                cur_batchsize = batch[0].size(0) # return 有行数，即有几句话
                b_input_ids = batch[0] # .to(device)
                raw_embeddings = model.roberta.embeddings.word_embeddings(b_input_ids) # 输入word_embedding
                inputs_embeds = raw_embeddings

                inputs_embeds = inputs_embeds # .to(device)
                attention_mask = batch[2] # .to(device)
                token_type_ids = batch[1] # .to(device)

                model.zero_grad()

                logits = model( # 开始训练
                    inputs_embeds=inputs_embeds, # 借用了RobertaForMaskedLM.roberta.embeddings.word_embeddings(input_ids)的word_embeddings
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )[0] # What does 0 mean? loss or logits(没提供label返回logits，即预测的y
                mlm_labels = batch[4]
                logits = logits # .to("cpu")
                logits_all = logits[mlm_labels > 0].view(cur_batchsize, 1, -1) # 在pytorch中view函数的作用为重构张量的维度————取mask位置的预测值

                logits_all = logits_all[:, 0, :] # 三维降两维
                save_logits.append(logits_all)
                
                labels = batch[3] # label，即class
                print(labels)

                # logits_all = self.cal_pred_use_logits(logits_all)

                loss = criterion(logits_all, labels)
                print("loss: ", loss)
                

                total_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                criterion.zero_grad()

            avg_train_loss = total_loss / len(args.train_dataloader)
            print("avg_train_loss : ")
            print(avg_train_loss)

            criterion_params = criterion.state_dict()["learnable_weights_for_label_wrods"]
            # print(criterion_params)

            if args.do_valid == 1:
                self.__valid(args)

            save_parameters(model, criterion_params, criterion)

        # save_parameters(model, criterion_params, criterion)


            
            # 保存logits值
            # save_logits = [i for item in save_logits for i in item]
            # print("save logits: ")
            # torch.save(save_logits, "./my_logits.pt") #将字符串写入文件中
            # print("write finished!")


    
    def __valid(self, args): # valid的目的是确认k_ratio
        self.criterion.eval()
        self.model.eval()
        all_valid_labels, all_valid_preds = self.__model_forward_valid(args.valid_dataloader)

        # print(len(all_valid_logits))
        # print(len(all_valid_logits[0]))
        # print(all_valid_logits[0])

        if self.args.criterion_type=='rank':
            best_k = self.criterion.determine_k(all_valid_preds, labels=all_valid_labels)
            # self.criterion.set_k_for_ranking(topk_ratio=best_k)
    
    def __model_forward_valid(self, dataloader):
        all_labels = []
        all_logits_all = None
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                print("======== The valid batch step is {%d} ========" % step)
                cur_batchsize = batch[0].size(0) # return 有行数，即有几句话
                raw_embeddings = self.model.roberta.embeddings.word_embeddings(batch[0]) # 输入word_embedding
                inputs_embeds = raw_embeddings

                self.model.zero_grad()

                logits = self.model( # 开始训练
                    inputs_embeds=inputs_embeds, # 借用了RobertaForMaskedLM.roberta.embeddings.word_embeddings(input_ids)的word_embeddings
                    attention_mask=batch[2],
                    token_type_ids=batch[1]
                )[0] # What does 0 mean? loss or logits(没提供label返回logits，即预测的y
                mlm_labels = batch[4]
                logits_all = logits[mlm_labels > 0].view(cur_batchsize, 1, -1)
                logits_all = logits_all[:, 0, :] # 三维降两维


                print("---------------:  ", type(logits_all))
                print("-----size: ", logits_all.size())
                
                if step == 0:
                    all_logits_all = logits_all
                else:
                    all_logits_all = torch.cat((all_logits_all, logits_all), 0)
                

                
                all_labels.append(batch[3])  # 加入labels
                

        
        all_labels = [i for per in all_labels for i in per]
        


        return all_labels, all_logits_all








if __name__ == "__main__":



    if args.task == "train_and_save_embedding_visualization":
        # print("task: train_and_save_embedding_visualization")
        runner = Runner(args)
        runner.train_model(args)

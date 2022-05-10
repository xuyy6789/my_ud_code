from turtle import forward
import torch
import torch.nn as nn
import tqdm
import xlwt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, accuracy_score

from configs import get_args_parser, get_model_classes
from criterion import get_criterion_classes
from datasets import Datasets

args = get_args_parser()

class TestAfterTrain(nn.Module):
    def __init__(self):
        super(TestAfterTrain, self).__init__()
        pass

    def forward(self):
        self.__load_model()
        self.__get_testdataset()
        self.__load_criterion()
        self.__model_forward_preds(args.test_dataloader)

    def __load_model(self):
        model_class = get_model_classes()[args.model_type]
        model = model_class['model'].from_pretrained("model_save")
        self.model = model

    def __load_criterion(self):
        criterion_class = get_criterion_classes()[args.criterion_type]
        criterion = criterion_class(args=args, encode_all_label_words=self.encode_all_label_words)
        criterion.load_state_dict(torch.load("./model_save/criterion.ckpt"))
        self.criterion = criterion
        print(criterion.state_dict())

    def __get_testdataset(self):
        print("begin to load test dataset")
        self.datasetclass = Datasets[args.dataset]
        self.datasetclass.get_template(args)
        self.datasetclass.get_testset_from_csv(self, args) # 得到了args.test_dataloader
        self.datasetclass.get_label_words(self, args) # 存入self.encode_all_label_words
        print("load finished!")
    
    

    def __predict_batch(self, all_logits):

        my_logits = self.criterion.get_label_words_mask_logits_singletoken(all_logits) # 获取mask向量中，对应label words的那些值
        weight = nn.functional.softmax(self.criterion.learnable_weights_for_label_wrods, dim=-1).reshape(-1)
        my_logits = weight * my_logits
        my_logits = nn.functional.softmax(my_logits, dim=-1)
        my_logits = torch.log(my_logits + 1e-15)
        scores = my_logits.reshape([my_logits.size(0), self.criterion.class_num, self.criterion.max_len])
        scores = torch.sum(scores, dim=-1)
        preds = torch.argmax(scores, dim=-1)

        return preds
        
    def __model_forward_preds(self, dataloader, dataloadertype='Test'):
        all_labels = []
        all_preds = []

        self.model.eval()
        self.criterion.eval()

        self.model.to(device)

        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                print("======== The batch step is {%d} ========" % step)
                cur_batchsize = batch[0].size(0) # return 有行数，即有几句话
                raw_embeddings = self.model.roberta.embeddings.word_embeddings(batch[0].to(device)) # 输入word_embedding
                inputs_embeds = raw_embeddings

                self.model.zero_grad()

                logits = self.model( # 开始训练
                    inputs_embeds=inputs_embeds.to(device), # 借用了RobertaForMaskedLM.roberta.embeddings.word_embeddings(input_ids)的word_embeddings
                    attention_mask=batch[2].to(device),
                    token_type_ids=batch[1].to(device)
                )[0] # What does 0 mean? loss or logits(没提供label返回logits，即预测的y
                mlm_labels = batch[4]
                logits = logits.to("cpu")
                logits_all = logits[mlm_labels > 0].view(cur_batchsize, 1, -1)
                logits_all = logits_all[:, 0, :] # 三维降两维

                all_labels.append(batch[3])  # 加入labels
                
                preds = self.__predict_batch(logits_all)
                all_preds.append(preds)
        #         preds = self.criterion.evaluate_batch(logits_all, self.prior_logits, self.tokenizer)
        #         all_preds.append(preds)
                print(all_preds)
                print(len(all_preds))
                print(all_labels)
                print(len(all_labels))
                

        all_labels = [i for per in all_labels for i in per]
        all_preds = [i for per in all_preds for i in per]


        # 保存logits值
#         print("save results: ")
#         torch.save(all_labels, "./result/result_labels_ranking_weights_kratio0.3.pt") #将字符串写入文件中
#         torch.save(all_preds, "./result/result_preds_ranking_weights_kratio0.3.pt") #将字符串写入文件中
#         print("write finished!")
        

        acc = accuracy_score(all_labels, all_preds)
        print("整体精确度acc：", acc)
        f1_micro_k = f1_score(all_labels, all_preds, average='micro')
        print("整体micro-f1值：", f1_micro_k )
        pre, rec, f1, sup = precision_recall_fscore_support(all_labels, all_preds)
        print("precision:", pre, "\nrecall:", rec, "\nf1-score:", f1, "\nsupport:", sup)



if __name__ == "__main__":

    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda:0")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    test_object = TestAfterTrain()
    test_object()
    
    




    
    


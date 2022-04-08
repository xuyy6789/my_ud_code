from sklearn.model_selection import PredefinedSplit
import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

class RankingCriterion(nn.Module): #

    def __init__(self, args, encode_all_label_words):
        super(RankingCriterion, self).__init__()
        self.args = args
        self.__get_encode_all_label_words_singletoken(encode_all_label_words) # 设置 self.encode_all_label_words

        

    def forward(self, all_logits, labels):


        my_logits = self.get_label_words_mask_logits_singletoken(all_logits) # 获取mask向量中，对应label words的那些值
        print("my_logits: ", my_logits.size())
        weight = nn.functional.softmax(self.learnable_weights_for_label_wrods, dim=-1).reshape(-1)
        my_logits = weight * my_logits
        my_logits = nn.functional.softmax(my_logits, dim=-1)
        my_logits = torch.log(my_logits + 1e-15)
        scores = my_logits.reshape([my_logits.size(0), self.class_num, self.max_len])
        scores = torch.sum(scores, dim=-1)
        my_criterion = torch.nn.CrossEntropyLoss()
        myloss = my_criterion(scores, labels)
    
        return myloss


    def get_label_words_mask_logits_singletoken(self, all_logits):
        """Return Tensor([[The predicted probability of label words of all classes by LM on mask position], [], [], ..., [Sentence 16]]), 16 is batch size"""
        label_words_logits = all_logits[:, self.encode_all_label_words_tensor.reshape(-1)] # reshape(-1): 展开成一行，即[[],[]]变成[]
        return label_words_logits
    
    def __get_encode_all_label_words_singletoken(self, encode_all_label_words):
        """set self.encode_all_label_words_tensor"""
        truncate_encode_all_label_words = [[word[0] for word in per_label_label_words] for per_label_label_words in encode_all_label_words] # 不是手动不要，而是直接裁剪encode
        
        class_num = len(truncate_encode_all_label_words) # class即label
        max_len = max([len(x) for x in truncate_encode_all_label_words])

        self.class_num = class_num
        self.max_len = max_len

        print("=====class num is: ", class_num)

        truncate_encode_all_label_words_tensor = torch.zeros([class_num, max_len], dtype=torch.long) # 让所有类的label words对其，不够的地方为0
        prompt_label_words_mask = torch.zeros_like(truncate_encode_all_label_words_tensor)
        multi_index_label_words = torch.zeros([class_num, max_len * class_num], dtype=torch.float)

        for id, per_class in enumerate(truncate_encode_all_label_words):
            truncate_encode_all_label_words_tensor[id, :len(per_class)] = torch.tensor(per_class).to(torch.long) #只起到赋值作用，把list的值赋值到tensor
            multi_index_label_words[id, id * max_len : (id * max_len + len(per_class))] = 1.0/max_len # type为tensor

        self.encode_all_label_words_tensor = truncate_encode_all_label_words_tensor
        self.prompt_label_words_mask = prompt_label_words_mask
        self.multi_index_label_words = multi_index_label_words

        
        self.num_all_label_words = sum([len(x) for x in truncate_encode_all_label_words]) # total number of label words

        learnable_weights_for_label_wrods = torch.zeros([class_num, max_len])
        # torch.nn.parameter.Parameter：
        # 可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面(net.parameter()
        # 中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        learnable_weights_for_label_wrods = torch.nn.parameter.Parameter(learnable_weights_for_label_wrods, requires_grad=True)
        self.learnable_weights_for_label_wrods = learnable_weights_for_label_wrods
    

    def set_k_for_ranking(self, topk_ratio=0):
        if topk_ratio > 0:
            self.topk = torch.tensor(int(self.num_all_label_words * topk_ratio)) # 选几个词

    def determine_k(self, all_logits, labels):
        """其中，logits是：已经获取mask向量中，对应label words的那些值，且已进行过normalization"""
        all_topk_ratio = [0.02 * i for i in range(1,26)]
        all_preds = []
        all_f1_micro = []
        all_precision_recall = []
        for k in all_topk_ratio:
            self.set_k_for_ranking(topk_ratio=k)
            preds = self.__predict(all_logits, k)
            if labels is not None:
                f1_micro_k = f1_score(labels, preds, average='micro')
                recall_k = precision_recall_fscore_support(labels, preds)
                all_f1_micro.append(f1_micro_k)
                all_precision_recall.append(recall_k)
            all_preds.append(preds)
        if len(all_f1_micro)>0:
            all_f1_micro = np.array(all_f1_micro)
            max_mic = np.max(all_f1_micro)
            best_ids = np.where(all_f1_micro==max_mic)[0]
            best_id = best_ids[0] ## get the first middle point
            # best_mic =  mics[best_id]
            best_k = all_topk_ratio[best_id]
            best_precision_recall = all_precision_recall[best_id]
            print("------best_k : ", best_k)
            print("------best_k 的 max_mic: ", max_mic)
            print("------best_precision_recall: ", best_precision_recall)
            print(all_f1_micro)
            return best_k

    # weight方法 or logits方法
    def __predict(self, all_logits, k_ratio):
        logits = self.get_label_words_mask_logits_singletoken(all_logits) # 获取mask向量中，对应label words的那些值
        logits = nn.functional.softmax(logits, dim=-1)
        logits = torch.log(logits + 1e-15)

        
       
        weight = [1 for i in range(len(self.learnable_weights_for_label_wrods.reshape(-1)))]
        weight = torch.tensor(weight)
        self.weights = weight


       
        self.set_k_for_ranking(k_ratio)
       
        nottopk = torch.argsort(-logits, dim=-1)[:,self.topk:]

        rowid = torch.arange(nottopk.size(0)).unsqueeze(-1).expand_as(nottopk)
       

        index = (rowid.reshape(-1), nottopk.reshape(-1))

        
        

        scores = torch.clone(self.weights).unsqueeze(0).repeat(logits.size(0),1)
      

        scores[index] = 0
        scores = scores.reshape([logits.size(0), self.class_num, self.max_len])
        print(scores.size())
        scores = torch.sum(scores, dim=-1)
        print(scores.size())
        preds = torch.argmax(scores, dim=-1)

        return preds





_CRITERION_CLASSES = {
    'rank': RankingCriterion
}

def get_criterion_classes():
    return _CRITERION_CLASSES

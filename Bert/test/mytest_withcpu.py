import sys
sys.path.append("../")

from model import BertForSentenceClassification
from model import BertConfig
from utils import LoadSingleSentenceClassificationDataset
from utils import logger_init
from transformers import BertTokenizer
import logging
import torch
import os
import time


class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'SingleSentenceClassification')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_chinese")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_file_path = os.path.join(self.dataset_dir, 'toutiao_train.txt')
        self.val_file_path = os.path.join(self.dataset_dir, 'toutiao_val.txt')
        self.test_file_path = os.path.join(self.dataset_dir, 'toutiao_test.txt')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.split_sep = '_!_'
        self.is_sample_shuffle = True
        self.batch_size = 64
        self.max_sen_len = None
        self.num_labels = 15
        self.epochs = 1
        self.model_val_per_epoch = 2
        logger_init(log_file_name='single', log_level=logging.INFO,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # 把原始bert中的配置参数也导入进来
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"###  {key} = {value}")

def evaluate(data_iter, model, device, PAD_IDX):
    model.eval()
    with torch.no_grad():
        print(data_iter)
        labels = []
        evaluate_time_start = time.time()
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            padding_mask = (x == PAD_IDX).transpose(0, 1)
            logits = model(x, attention_mask=padding_mask)
            # 找标签
            print("DEBUG: ", [tensor_.item() for tensor_ in logits.argmax(1)])
            labels += [tensor_.item() for tensor_ in logits.argmax(1)]
        evaluate_time_end = time.time()
        model.train()
        res = {
            "label": [tensor_.item() for tensor_ in logits.argmax(1)],
            "time": evaluate_time_end - evaluate_time_start
        }
        return res


if __name__ == "__main__":
    config = ModelConfig()
    model = BertForSentenceClassification(config,
                                          config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path, map_location=torch.device('cpu'))
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行预测......")
    model = model.to(config.device)
    data_loader = LoadSingleSentenceClassificationDataset(vocab_path=config.vocab_path,
                                                          tokenizer=BertTokenizer.from_pretrained(
                                                              config.pretrained_model_dir).tokenize,
                                                          batch_size=config.batch_size,
                                                          max_sen_len=config.max_sen_len,
                                                          split_sep=config.split_sep,
                                                          max_position_embeddings=config.max_position_embeddings,
                                                          pad_index=config.pad_token_id,
                                                          is_sample_shuffle=config.is_sample_shuffle)
    # train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(config.train_file_path,
    #                                                                        config.val_file_path,
    #                                                                        config.test_file_path)
    # 只读一句话，而不是读测试集
    test_iter = data_loader.load_one_data("predict.txt")
    evaluate(test_iter, model, device=config.device, PAD_IDX=data_loader.PAD_IDX)
    os.remove("predict_None.pt")
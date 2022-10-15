import random
import sys

sys.path.append("Bert")

import predict
from PyQt5.QtWidgets import QApplication, QDialog
import time
import os
import torch
import logging
import random
from FastText import testFastText
from Bert.test import mytest


# 映射字典
id2name_en = {
    0: "story",
    1: "culture",
    2: "entertainment",
    3: "sports",
    4: "finance",
    5: "house",
    6: "car",
    7: "edu",
    8: "tech",
    9: "military",
    10: "travel",
    11: "world",
    12: "stock",
    13: "agriculture",
    14: "game"
    }

id2name_ch = {
    0: "民生 故事",
    1: "文化 文化",
    2: "娱乐 娱乐",
    3: "体育 体育",
    4: "财经 财经",
    5: "房产 房产",
    6: "汽车 汽车",
    7: "教育 教育",
    8: "科技 科技",
    9: "军事 军事",
    10: "旅游 旅游",
    11: "国际 国际",
    12: "证券 股票",
    13: "农业 三农",
    14: "电竞 游戏"
}

name_en2name_ch = {
"story": "民生 故事",
"culture": "文化 文化",
"entertainment": "娱乐 娱乐",
"sports": "体育 体育",
"finance": "财经 财经",
"house": "房产 房产",
"car": "汽车 汽车",
"edu": "教育 教育",
"tech": "科技 科技",
"military": "军事 军事",
"travel": "旅游 旅游",
"world": "国际 国际",
"stock": "证券 股票",
"agriculture": "农业 三农",
"game": "电竞 游戏"
}


# 顶层模块
class MainDialog(QDialog):
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)
        self.ui = predict.Ui_Dialog()
        self.ui.setupUi(self)
        self.predict_text = None
        self.use_fasttext = False
        self.use_bert = False
        self.acc = 0.0
        self.evaluate_time_start = None
        self.ui.textBrowser_4.setText(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} 启动完成")

    # 获取输入框内容
    def setInput(self, text):
        print("Input:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
              text)
        self.predict_text = text

    # 调用FastText和Bert进行预测
    def predict(self):
        print("PREDICT")
        # 如果输入内容为空
        if self.predict_text is None or self.predict_text == "":
            print("\tERROR: Text to predict is empty")
            self.ui.textBrowser_4.setText(f"ERROR: Text to predict is empty")
            return
        else:
            try:
                if self.use_fasttext:
                    self.evaluate_time_start = time.time()
                    with open("predict_text.txt", "w", encoding="utf-8") as f:
                        f.write(self.predict_text)
                    try:
                        stopwords = testFastText.load_stopwords("FastText/data/stopwords.txt")
                        res = testFastText.predict("FastText/model/2022-10-05-14-05-17.bin", stopwords, "predict_text.txt")
                    except Exception as e:
                        print("\tERROR:", e)
                        return
                    else:
                        # 正常处理
                        evaluate_time_end = time.time()
                        print(res)
                        self.ui.textBrowser.setText(res['label'].replace('news_', ''))
                        self.ui.textBrowser_2.setText(f"{evaluate_time_end - self.evaluate_time_start:.4f}s")
                elif self.use_bert:
                    self.evaluate_time_start = time.time()
                    try:
                        with open("predict_text.txt", "w", encoding="utf-8") as f:
                            f.write(self.predict_text + "_!_0")
                        # 设置参数
                        config = mytest.ModelConfig()
                        model = mytest.BertForSentenceClassification(config,
                                                              config.pretrained_model_dir)
                        model_save_path = os.path.join(config.model_save_dir, 'model.pt')
                        # 载入已有模型
                        if os.path.exists(model_save_path):
                            loaded_paras = torch.load(model_save_path, map_location=torch.device('cpu'))
                            model.load_state_dict(loaded_paras)
                            logging.info("## 成功载入已有模型，进行预测......")
                        model = model.to(config.device)
                        # 加载数据
                        data_loader = mytest.LoadSingleSentenceClassificationDataset(vocab_path=config.vocab_path,
                                                                              tokenizer=mytest.BertTokenizer.from_pretrained(
                                                                                  config.pretrained_model_dir).tokenize,
                                                                              batch_size=config.batch_size,
                                                                              max_sen_len=config.max_sen_len,
                                                                              split_sep=config.split_sep,
                                                                              max_position_embeddings=config.max_position_embeddings,
                                                                              pad_index=config.pad_token_id,
                                                                              is_sample_shuffle=config.is_sample_shuffle)
                        test_iter = data_loader.load_one_data("predict_text.txt")
                        # 进行测试
                        res = mytest.evaluate(test_iter, model, device=config.device, PAD_IDX=data_loader.PAD_IDX)
                        # 删除临时文件
                        os.remove("predict_text_None.pt")
                    except Exception as e:
                        print("\tERROR:", e)
                        self.ui.textBrowser_4.setText(f"ERROR: {e}")
                    else:
                        try:
                            # 正常处理
                            evaluate_time_end = time.time()
                            print(res)
                            self.ui.textBrowser.setText(id2name_en[res['label'][0]])
                            self.ui.textBrowser_2.setText(f"{float(evaluate_time_end - self.evaluate_time_start):.4f}s")
                        except Exception as e:
                            print("\tERROR:", e)
                            self.ui.textBrowser_4.setText(f"ERROR: {e}")
                else:
                    raise RuntimeError("Haven't Select Model [FastText/Bert]")
            except RuntimeError:
                # 如果没有选择模型
                print("\tERROR: Haven't Select Model [FastText/Bert]")
                self.ui.textBrowser_4.setText(f"ERROR: Haven't Select Model [FastText/Bert]")
                return
            else:
                # 提示输出结果
                if self.use_fasttext:
                    self.ui.textBrowser_4.setText(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} FastText 预测完成")
                elif self.use_bert:
                    self.ui.textBrowser_4.setText(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} Bert 预测完成")
                else:
                    self.ui.textBrowser_4.setText(f"ERROR: Text to predict is empty")

    # 设置使用FastText模型
    def setFastText(self, res):
        print("SET FastText", res)
        self.use_fasttext = res
        # 准确率为测试结果，参照Readme:安装步骤/FastText/评估
        self.ui.textBrowser_3.setText("准确率: 0.906")

    # 设置使用Bert模型
    def setBert(self, res):
        print("SET Bert", res)
        self.use_bert = res
        # 准确率为测试结果，参照Readme:安装步骤/Bert/评估
        self.ui.textBrowser_3.setText("准确率: 0.890")

    # 从测试集随机选择新闻标题
    def randomSentence(self):
        with open("Bert/data/SingleSentenceClassification/toutiao_test.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            max_num = len(lines)
            line_index = random.randint(0, max_num-1)
            text = lines[line_index].split('_!_')[0]
            self.predict_text = text
            self.ui.lineEdit.setText(text)

if __name__ == '__main__':
    myapp = QApplication(sys.argv)
    myDialog = MainDialog()
    myDialog.show()
    sys.exit(myapp.exec_())
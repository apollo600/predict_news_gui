## 配置环境

```shell
cd FastText
pip install .
cd ..
pip install -r requirements.txt
```

## 安装步骤

### FastText

实验训练条件为`Windows`

```
cd FastText
# 训练
python testFastText.py --train
# 评估
python testFastText.py --eval --model=model\2022-10-15-14-56-15.bin
# 推理
python testFastText.py --predict=..\predict_text.txt --model=model\2022-10-15-14-56-15.bin
```

### Bert

实验训练条件为`NVIDIA GeForce GTX 1080 Ti` `Ubuntu 18.04.1`

```shell
cd Bert
# 训练&评估
cd Tasks
python TaskForSingleSentenceClassification.py
# 推理
## 需要使用CPU
cd ../test
python mytest_withcpu.py
## 使用GPU
cd ../test
python mytest.py
```

## demo命令

### 启动GUI

```shell
python mainDialog.py
```

### 使用说明：

1. 输入要预测的新闻标题，或点击随机短标题，会从测试集中随机选择一条标题
2. 选择要使用的模型，FastText/Bert
3. 点击开始预测，稍后下方出现预测结果

### 用户交互

#### 异常处理

1. 未输入新闻标题

   ![image-20221015155359753](https://raw.githubusercontent.com/apollo600/images/main/20221015155359.png)

2. 未选择预测模型

   ![image-20221015155434219](https://raw.githubusercontent.com/apollo600/images/main/20221015155434.png)

### 界面设计

按照操作前后顺序，从上往下排布区域，输入标题->选择模型->开始预测->输出结果

右下角设置为操作提示区，如启动完成、预测完成、错误提示等
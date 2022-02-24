# 1.使用中文 bert nsp 试一下
## train
```buildoutcfg
allennlp train scripts/my_text_classifier_bert.jsonnet -s checkpoint --include-package my_modules -f
```
## predict
```buildoutcfg
allennlp predict checkpoint/model.tar.gz data/chinese_question_sim/test.csv --output-file data/chinese_question_sim/predict_result.csv --include-package my_modules --predictor sentence_sim_classifier --batch-size 8 --silent
```

# todo: 
    1、使用robert试一下【1 input 和2 input 都试一下】  【todo】
    2、bert 进行与train test 预训练，然后再 fine tune 【doing】
    3、参考 https://blog.csdn.net/daniellibin/article/details/118059267?spm=5176.21852664.0.0.16ef2448aqJG2V 做数据tric   
    4、可以试试新的 robert zh : F:\home\featurize\data\RoBERTa_zh_L12_PyTorch


# 2.robert 使用 robert_pretrain 1 个 input
## train 
```buildoutcfg
allennlp train scripts/my_text_classifier_robert.jsonnet -s checkpoint --include-package my_modules -f
```

# 3.bert 使用 8 num_hidden_layers
需要修改`chinese_question_bert_base/config.json`中的`"num_hidden_layers": 8,`
## train
```buildoutcfg
allennlp train scripts/my_text_classifier_bert.jsonnet -s checkpoint --include-package my_modules -f
```
 {
  "best_epoch": 1,
  "best_validation_accuracy": 0.89,
  "best_validation_loss": 0.2764296330038517
}
## predict
```buildoutcfg
allennlp predict checkpoint/model.tar.gz data/chinese_question_sim/test.csv --output-file data/chinese_question_sim/predict_result.csv --include-package my_modules --predictor sentence_sim_classifier --batch-size 8 --silent
```
test acc 0.8876


# 4. bert 2 input + layer 8 + sent_att: 乘以w之后，再sum
## train
```buildoutcfg
allennlp train scripts/my_text_classifier_bert_sent.jsonnet -s checkpoint --include-package my_modules -f
```
{
  "best_validation_accuracy": 0.666,
  "best_validation_loss": 0.6106449630260468
}
## bert 2 input + layer 8 + sent_att: concat 
 {
  "best_validation_accuracy": 0.655,
  "best_validation_loss": 0.629179696559906
}
##  bert 2 input + layer 8 + sent_att 乘以w之后，再concat
{
  "best_validation_accuracy": 0.649,
  "best_validation_loss": 0.6213608934879303
}


# todo:         8 layer
0、bert 自定义  
训练3个epoch， 最好的是2个epoch, train acc 是0.904， vila acc 是0.888 
test acc 0.8882

1、bert 最后一层向量取平均+与最后一层pool拼接
训练3个epoch， 最好的是2个epoch, train acc 是0.914， vila acc 是0.89
test acc  0.8802

2、bert 最后一层向量取平均【结论，调整lr 没有用，还是】
  lr:1e-5 训练3个epoch， 最好的是1epoch, train acc 是0.916， vila acc 是0.893
    test acc  
  lr:4e-5 训练5个epoch， 最好的是3epoch, train acc 是0.916， vila acc 是0.891
    test acc
  lr:2e-5 训练5个epoch， 最好的是0epoch, train acc 是0.817， vila acc 是0.892
    test acc
  lr:2e-5 12 layer 训练5个epoch， 最好的是 0epoch, train acc 是 0.799， vila acc 是 0.898
    test acc 0.8864

3、bert 最后一层向量进行sent attention 
  lr:2e-5 8 layer 训练5个epoch， 最好的是 0 epoch, train acc 是 0.802， vila acc 是 0.879
    test acc  0.8736


4、bert 最后一层向量进行lstm+sent_attention
  lr:1e-5 8 layer 训练5个epoch， 最好的是 0 epoch, train acc 是 0.915， vila acc 是 0.89
    test acc  0.881


5、 使用 nezha-chinese-base 最后一层向量进行lstm+sent_attention
    lr:1e-4 12 layer 训练5个epoch， 最好的是   epoch, train acc 是  ， vila acc 是  
    test acc  

  结论：这里的nezha因为是没有在本语料上进行ITPT（任务内预训练），所以lr要设置的大一些：1e-4 才能收敛的快一点
                                                        lr: 1e-5,收敛很慢
  
  注意：BERT-ITPT-FiT 的意思是“BERT + with In-Task Pre-Training + Fine-Tuning”

todo: 1、怎样保存并查看 学习曲线，loss, tenserbord
      2、有一个问题，best model 的 train acc 没有，需要从最后结果看到best vali acc，
         然后返回去找对应epoch的train acc

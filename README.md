emmmm就算是个搬运工和整合工啦，具体原理就不介绍了，懂的都懂哈哈哈，写这篇文章主要是为了大家更方便调用，目前感觉整个代码还是比较好操作的，具体的代码可以自行参考XXX，都比较简单适合BERT的初学者

# 环境准备

1.安装bert-for-tf2
```
pip install bert-for-tf2
```

2.从 https://github.com/google-research/bert 下载需要的模型

* 以 https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip 这个模型为例，将其解压后的文件夹置于BERTClassifier.py的同一文件夹下

* 这个数据集包含了很多unused字符，因此我将第101行-103行改为了：
```
101 [UNK]
102 [CLS]
103 [SEP]
```

# 数据准备

* 数据集至少应包含文本列和分类列,本文选择了 https://github.com/skdjfla/toutiao-text-classfication-dataset 的头条文本分类数据


```python
import pandas as pd
corpusDf=pd.read_csv("SampleData/toutiao_cat_data.txt",sep="_!_",encoding="utf8").sample(50)#为了节省时间我就只采样5000条咯
```


```python
corpusDf.drop(["newsId","newsNum","loc"],axis=1,inplace=True)
```


```python
corpusDf=corpusDf.loc[:,["text","class"]]
corpusDf.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>289273</th>
      <td>中国男性娶妻标准汇总</td>
      <td>news_entertainment</td>
    </tr>
    <tr>
      <th>247864</th>
      <td>10省市确定上调最低工资标准！你的工资涨了吗？</td>
      <td>news_finance</td>
    </tr>
    <tr>
      <th>145052</th>
      <td>蓝田鲍旗寨为什么被网友称之为中国的普罗旺斯？</td>
      <td>news_travel</td>
    </tr>
    <tr>
      <th>185919</th>
      <td>温暖的弦里温暖的姐姐温柔怎么这么眼熟，原来是《琅琊榜》里的她</td>
      <td>news_entertainment</td>
    </tr>
    <tr>
      <th>194963</th>
      <td>林允出席某活动，网友：星爷挑女主角的眼光果然让人赞叹呀！</td>
      <td>news_entertainment</td>
    </tr>
  </tbody>
</table>
</div>



* text列应以list形式呈现并在开头与结尾加上"\[CLS\]"和"\[SEP\]"，并记录下最长句子


```python
corpusDf["text"]=corpusDf["text"].apply(lambda row:["[CLS]"]+list(row)+["[SEP]"])
maxLen=max([len(row) for row in corpusDf["text"]])
print("最大长度：{}".format(maxLen))
```

    最大长度：37
    


```python
corpusDf.reset_index(drop=True)["text"][0]
```




    ['[CLS]', '中', '国', '男', '性', '娶', '妻', '标', '准', '汇', '总', '[SEP]']



* 类别应变为[0,1,2,...]的格式，同时记录下类别的数目


```python
classTuple=tuple(set(corpusDf["class"].values.tolist()))
classNum=len(classTuple)
print("分类数目：{}".format(classNum))
```

    分类数目：12
    


```python
classTuple
```




    ('news_edu',
     'news_sports',
     'news_world',
     'news_car',
     'news_travel',
     'news_military',
     'news_agriculture',
     'news_finance',
     'news_culture',
     'news_tech',
     'news_entertainment',
     'news_game')




```python
corpusDf["class"]=corpusDf["class"].apply(lambda row:classTuple.index(row))
```


```python
corpus=corpusDf.values
```

# 区分训练集和测试集


```python
from sklearn.model_selection import train_test_split

trainX,testX,trainY,testY=train_test_split(corpus[:,0],corpus[:,1],test_size=0.3)
```

# 构造模型


```python
from BERTClassifier import MyBERTClassier
```


```python
myModel=MyBERTClassier(classNum,XMaxLen=maxLen,learning_rate=0.0001)
```

    21128it [00:00, 1626601.60it/s]
    

# 训练模型


```python
myModel.fit(trainX,trainY,epochs=15,batch_size=32)
```

    Epoch 1/3
    2/2 [==============================] - 0s 142ms/step - loss: 0.8792
    Epoch 2/3
    2/2 [==============================] - 0s 139ms/step - loss: 0.5649
    Epoch 3/3
    2/2 [==============================] - 0s 145ms/step - loss: 0.3153
    Epoch 1/12
    1/1 [==============================] - 0s 1000us/step - loss: 0.1388
    Epoch 2/12
    1/1 [==============================] - 0s 0s/step - loss: 0.0184
    Epoch 3/12
    1/1 [==============================] - 0s 999us/step - loss: 0.0126
    Epoch 4/12
    1/1 [==============================] - 0s 1ms/step - loss: 0.0135
    Epoch 5/12
    1/1 [==============================] - 0s 2ms/step - loss: 0.0142
    Epoch 6/12
    1/1 [==============================] - 0s 1000us/step - loss: 0.0103
    Epoch 7/12
    1/1 [==============================] - 0s 1000us/step - loss: 0.0086
    Epoch 8/12
    1/1 [==============================] - 0s 1000us/step - loss: 0.0075
    Epoch 9/12
    1/1 [==============================] - 0s 1000us/step - loss: 0.0074
    Epoch 10/12
    1/1 [==============================] - 0s 1ms/step - loss: 0.0085
    Epoch 11/12
    1/1 [==============================] - 0s 1000us/step - loss: 0.0077
    Epoch 12/12
    1/1 [==============================] - 0s 0s/step - loss: 0.0067
    

# 预测数据


```python
import numpy as np

np.argmax(myModel.predict(testX),axis=-1)
```




    array([ 9, 10,  0,  0,  9,  9,  9,  1,  9,  9,  2,  3,  1, 10,  0],
          dtype=int64)



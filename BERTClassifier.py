import numpy as np
import pandas as pd
import os
import tqdm
import bert
from tensorflow import keras
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
from sklearn.preprocessing import OneHotEncoder
import pickle as pkl
from sklearn.metrics import f1_score

class MyBERTClassier:

    def __init__(self,
                classNum=2,
                preModelPath="chinese_L-12_H-768_A-12",
                learning_rate=0.1,
                XMaxLen=5,
            ):
        self.preModelPath=preModelPath
        self.learning_rate=learning_rate
        self.XMaxLen=XMaxLen
        self.classNum=classNum
        self.maxLen=XMaxLen
        self.buildVocab()
        self.tokenizer=bert.bert_tokenization.FullTokenizer(os.path.join(self.preModelPath, "vocab.txt"), do_lower_case=True)

        self.buildModel()

    def buildVocab(self):
        with open(os.path.join(self.preModelPath,"vocab.txt"),"r",encoding="utf8") as vocabFile:
            self.XVocabList=[row.strip() for row in tqdm.tqdm(vocabFile)]
            self.XVocabSize=len(self.XVocabList)

    def removeUNK(self,seqList):
        return [[wItem for wItem in row if wItem in self.XVocabList] for row in seqList]

    def buildModel(self):
        
        inputLayer = keras.layers.Input(shape=(self.maxLen,), dtype='int32')

        bert_params = bert.params_from_pretrained_ckpt(self.preModelPath)
        bertLayer = bert.BertModelLayer.from_params(bert_params, name="bert")(inputLayer)

        flattenLayer = keras.layers.Flatten()(bertLayer)
        outputLayer = keras.layers.Dense(
            self.classNum, activation="softmax")(flattenLayer)

        self.model = keras.models.Model(inputLayer,outputLayer)
        self.model.compile(loss="SparseCategoricalCrossentropy",
                            optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate))

    def fit(self,X,y,epochs=1,batch_size=64):
        '''
        X:cutted seq
        y:cutted y
        '''

        X=np.array([self.tokenizer.convert_tokens_to_ids([wItem for wItem in row if wItem in self.XVocabList]) for row in X])
        X=np.array([row+[0]*(self.maxLen-len(row)) if len(row)<self.maxLen else row[:self.maxLen] for row in X.tolist()])
        y=y.astype(np.int32)

        preEpochs=3
        self.model.fit(X,y,epochs=preEpochs,batch_size=batch_size)
        self.model.layers[1].trainable=False
        
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15)
        self.model.fit(X, y, epochs=epochs-preEpochs,
                       batch_size=3*batch_size, callbacks=[es])
        
    def predict(self,X):

        X=np.array([self.tokenizer.convert_tokens_to_ids([wItem for wItem in row if wItem in self.XVocabList]) for row in X])
        X=np.array([row+[0]*(self.maxLen-len(row)) if len(row)<self.maxLen else row[:self.maxLen] for row in X.tolist()])

        preY=self.model.predict(X)
        
        return preY

if __name__=="__main__":
    print("loading data ...")
    corpusDf=pd.read_csv("data/CMS-DATA.csv")
    corpusDf.dropna(inplace=True)
    corpusDf["text"]=corpusDf["text"].apply(lambda row:row.replace(" ","").strip())

    print("restructure Y (could be tagged) ...")
    classList=list(set(corpusDf["class"].values.tolist()))
    classNum=len(classList)
    corpusDf["class"]=corpusDf["class"].apply(lambda row:classList.index(row))

    print("prefix and suffix ...")
    corpusDf["text"]=corpusDf["text"].apply(lambda row:["[CLS]"]+list(row)+["[SEP]"])
    corpus=corpusDf.values

    print("splitting train/test ...")
    trainX,testX,trainY,testY=train_test_split(corpus[:,0],corpus[:,1],test_size=0.3,stratify=True)

    print("building model ...")
    seqList=corpus[:,0].tolist()
    maxLen=int(np.mean([len(row) for row in seqList]))
    print("max_len:",maxLen)
    myModel=MyBERTClassier(classNum,XMaxLen=maxLen,learning_rate=0.0001)
    print(myModel.model.summary())

    print("training model ...")
    myModel.fit(trainX,trainY,epochs=150,batch_size=32)

    print("testing model ...")
    preY=myModel.predict(testX)
    print("test f1:",f1_score(testY.astype(np.int32),np.argmax(preY,axis=-1),average="macro"))

    trainPreY=myModel.predict(trainX)
    print("train f1:",f1_score(trainY.astype(np.int32),np.argmax(trainPreY,axis=-1),average="macro"))

    print("saving model ...")
    myModel.model.save_weights("model/BERT-ClassifierModel")
    hpDict={
        "learning_rate":myModel.learning_rate,
        "maxLen":myModel.maxLen,
        "preModelPath":myModel.preModelPath,
        "tokenizer":myModel.tokenizer,
        "XVocabSize":myModel.XVocabSize,
        "classNum":myModel.classNum,
        "classList":classList
    }
    with open("model/BERTClassifier.pkl","wb+") as myModelFile:
        pkl.dump(hpDict,myModelFile)
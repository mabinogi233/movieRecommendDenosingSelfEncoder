import get_data

import cnnmodel

import os
import sys
# Path for spark source folder
os.environ['SPARK_HOME'] = r"D:\spark-3.2.0-bin-hadoop2.7\spark-3.2.0-bin-hadoop2.7"
os.environ['HADOOP_HOME'] = r"D:\hadoop-2.7.7\hadoop-2.7.7"
os.environ['JAVA_HOME'] = r"D:\jdk8"
os.environ['PYSPARK_PYTHON'] = "D:\Anaconda\python.exe"
os.environ['PYSPARK_DRIVER_PYTHON'] = "D:\Anaconda\python.exe"

# Append pyspark to Python Path
sys.path.append(r"D:\spark-3.2.0-bin-hadoop2.7\spark-3.2.0-bin-hadoop2.7\bin")
sys.path.append(r"D:\hadoop-2.7.7\hadoop-2.7.7\bin")
sys.path.append(r"D:\spark-3.2.0-bin-hadoop2.7\spark-3.2.0-bin-hadoop2.7\python")
sys.path.append(r"D:\spark-3.2.0-bin-hadoop2.7\spark-3.2.0-bin-hadoop2.7\python\lib\py4j-0.10.9.2-src.zip")


from pyspark import SparkContext

from pyspark import SparkConf

import time

import params

import numpy as np

import keras

#仅用于评估
from sklearn import metrics

#验证
def check(sc, old_ratings, max_movie_num, movie_list,model):
    true_y = get_data.batch_data_to_keras_train_rdd(sc, old_ratings, max_movie_num, movie_list)
    pre_y = model.predict(true_y)
    mse = metrics.mean_squared_error(true_y.astype('float64').reshape((true_y.shape[0],true_y.shape[2])),pre_y.astype('float64').reshape((true_y.shape[0],true_y.shape[2])))
    return mse

#继续训练，非从头开始训练
def train():
    conf = SparkConf().setAppName("dae").setMaster("local")
    sc = SparkContext(conf=conf.setAppName("dae"))
    #预加载
    old_ratings, max_movie_num, movie_list = get_data.pre_data_to_keras_train_rdd(sc)
    #非retrain，加载movieid索引表
    movie_list = get_data.load_movie_list()

    # 建立模型
    print("模型建立中")
    autoencoder = cnnmodel.build_keras_model()

    #加载参数
    cnnmodel.load_model(autoencoder, params.tag_ae)

    #训练epochs批次
    for i in range(params.epochs):
        if (i != 0):
            cnnmodel.load_model(autoencoder, params.tag_ae)
        #抽取batch数据
        train_rdd = get_data.batch_data_to_keras_train_rdd(sc, old_ratings, max_movie_num, movie_list)
        noise = np.clip(0.5 * np.random.normal(0.5, 0.09, train_rdd.shape), 0, 1)
        print("第",i + 1, "次训练：数据收集完毕")
        #每个epochs训练的次数
        for j in range(params.verbose):
            msg = autoencoder.train_on_batch(train_rdd+noise, train_rdd)
            if (j % (params.verbose) == 0):
                print("当前指标：",msg)
        #if(i%10==0):
            #mse = check(sc,old_ratings,max_movie_num,movie_list,autoencoder)
            #print(mse)
        cnnmodel.save_model(autoencoder, params.tag_ae)

#重新开始训练
def reTrain():
    conf = SparkConf().setAppName("dae").setMaster("local")
    sc = SparkContext(conf=conf.setAppName("dae"))
    # 预加载
    old_ratings, max_movie_num, movie_list = get_data.pre_data_to_keras_train_rdd(sc)
    # retrain，保存movieid索引表
    get_data.store_movie_list(movie_list)

    # 建立模型
    print("模型建立中")
    autoencoder = cnnmodel.build_keras_model()

    # 训练epochs批次
    for i in range(params.epochs):
        if (i != 0):
            cnnmodel.load_model(autoencoder, params.tag_ae)
        # 抽取batch数据
        train_rdd = get_data.batch_data_to_keras_train_rdd(sc, old_ratings, max_movie_num, movie_list)
        noise = np.clip(0.5 * np.random.normal(0.5, 0.09, train_rdd.shape), 0, 1)
        print("第",i + 1, "次训练：数据收集完毕")
        # 每个epochs训练的次数
        for j in range(params.verbose):
            msg = autoencoder.train_on_batch(train_rdd+noise, train_rdd)
            if (j % (params.verbose) == 0):
                print("当前指标：", msg)
        if (i % 10 == 0):
            mse= check(sc,old_ratings, max_movie_num, movie_list, autoencoder)
            print(mse)
        cnnmodel.save_model(autoencoder, params.tag_ae)

if __name__ == '__main__':
    train()
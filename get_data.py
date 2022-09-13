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

from pyspark.sql import SQLContext

import time

import numpy as np

import params

import random

#加载csv,使用spark.sqlContext，加载为spark.dataframe
def sprak_procuration_load_csv(file_path,sc):
    print("spark procuration start load: ", file_path)
    sqlContext = SQLContext(sc)
    sdf = sqlContext.read.csv(file_path)
    print("spark procuration load success: ", file_path)
    return sdf

#加载目录下的全部csv文件，并封装为spark可用的DataFrame，并加入字典，key为csv名称，value为其DataFrame
def load_any_data(data_root_path,sc):
    data_dict = {}
    if(not os.path.isdir(data_root_path)):
        return None
    for file in os.listdir(data_root_path):
        if(".csv" in file):
            start = time.time()
            print("start load: ",file)
            df = sprak_procuration_load_csv(data_root_path + os.sep + file,sc)
            data_dict[file.replace(".csv","")] = df
            end = time.time()
            print("load success: ",file,"used time: ",end-start,"s")
    return data_dict

#数据预加载
def pre_data_to_keras_train_rdd(sc):
    #读取ratings.csv，加载进入框架
    data_dict = load_any_data(params.data_file_root_path, sc)
    old_ratings = data_dict['ratings']

    #max_user_movie_num = old_ratings.withColumn("_c1_int", old_ratings['_c1'].cast('int')).select(max("_c1_int")).first()["max(_c1_int)"]
    #读取movies.csv，加载进入框架，剔除标题行
    old_movie = data_dict['movies'].filter("_c0!='movieId'")

    #生成movie_list
    movie_list = old_movie.rdd.map(lambda x:x[0]).distinct().collect()

    #max_movie_num = old_movie.withColumn("_c0_int", old_movie['_c0'].cast('int')).select(max("_c0_int")).first()[
    #    "max(_c0_int)"]

    #剔除标题行
    old_ratings = old_ratings.filter("_c0!='userId'")

    #返回ratings,movie个数，movieid组成的list
    return old_ratings,len(movie_list),movie_list

#随机抽取batch用户，生成一次训练的样本
def batch_data_to_keras_train_rdd(sc,old_ratings,max_movie_num,movie_list):

    #随机抽取batch个用户
    user_rdd = old_ratings.rdd.map(lambda x: x[0]).distinct()
    random_user_list = user_rdd.takeSample(True,params.batch_size)
    #训练集
    train_x = np.zeros((params.batch_size,params.input_shape))
    #生成sql查询条件
    cond = ""
    for user_i in range(len(random_user_list)):
        if(user_i!=0):
            cond += ' or '
        cond += "_c0=='"+random_user_list[user_i]+"'"

    print(cond)
    #pyspark sql查询
    one_train_rdd = old_ratings.filter(cond)
    #封装为训练集（分片邻接矩阵）
    for item in one_train_rdd.collect():
        if(float(item["_c2"])>=3):
            train_x[random_user_list.index((item["_c0"]))][movie_list.index((item["_c1"]))] = 1
    #reshape为2D卷积输入格式
    train_x = train_x.reshape((train_x.shape[0],1,train_x.shape[1],1))
    return train_x

#由于rdd数据顺序问题，需要固化第一次train生成的index-movieid的索引表
def store_movie_list(movie_list):
    random.shuffle(movie_list)
    with open(params.movie_list_file_path+os.sep+"movie_list.txt",'w',encoding="utf-8") as f:
        for item in movie_list:
            f.write(str(item))
            f.write("\n")

#加载 movieid索引表
def load_movie_list():
    movie_list = []
    with open(params.movie_list_file_path+os.sep+"movie_list.txt",'r',encoding="utf-8") as f:
        for item in f.readlines():
            movie_list.append(str(item.strip().replace("\n","")))
    return movie_list


#最大的k个数据的下标
def top_k_index(data,k):
    #key为元素，value为下标
    max_dict = {}
    for i in range(len(data)):
        item = data[i]
        if(len(max_dict.keys())<k):
            max_dict[item] = i
        else:
            #max_dict已满
            if(item>min(max_dict.keys())):
                del max_dict[min(max_dict.keys())]
                max_dict[item] = i
    list_value = []
    for k in sorted(max_dict.keys(),reverse = True):
        list_value.append(max_dict[k])

    if(params.debug):
        print("为1的概率:下标")
        print(max_dict)

    return list_value


import cnnmodel

import params

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

import get_data

import pymongo

import numpy as np

#预测待推荐的k部电影
def _prec(model,dict_x,movie_list,k):
    #dict_x为评分表，（movieid，rank）
    test_x = np.zeros((1,params.input_shape))
    for movie_id in dict_x.keys():
        if(float(dict_x[movie_id])>=3):
            test_x[0][movie_list.index(movie_id)] = 1
    test_x = test_x.reshape((1,1,params.input_shape,1))
    prec_y = model.predict(test_x)
    prec_y = prec_y +  np.clip(np.random.normal(0.25, 0.04, prec_y.shape), 0, 0.5)
    prec_y = list(prec_y.reshape((params.input_shape,)))[0:len(movie_list)]
    max_k_plus_list = get_data.top_k_index(list(prec_y),k+len(dict_x.keys()))

    prec_list_movie_id = []
    for movie_id_index in max_k_plus_list:
        movie_id = movie_list[movie_id_index]
        if(len(prec_list_movie_id)<k and movie_id not in dict_x.keys()):
            prec_list_movie_id.append(movie_id)

    if(params.debug):
        print("为1概率最大的k个元素的下标")
        print(max_k_plus_list)
        print("推荐的top-k电影的id")
        print(prec_list_movie_id)

    return prec_list_movie_id

def prec(phoneNum,k):
    #配置环境
    conf = SparkConf().setAppName("dae").setMaster("local")
    sc = SparkContext(conf=conf)

    dictx = load_rank(phoneNum)

    if(dictx==None):
        select_random(sc,k,phoneNum)
    else:
        # 预测模式，加载movieid索引表
        movie_list = get_data.load_movie_list()

        # 建立模型
        autoencoder = cnnmodel.build_keras_model()

        # 加载参数
        cnnmodel.load_model(autoencoder, params.tag_ae)

        #预测
        movie_id = _prec(autoencoder, dictx, movie_list, k)

        #输出
        save_result(phoneNum,movie_id,sc)

        sc.stop()

#加载rank
def load_rank(phoneNum):
    try:
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["movie"]
        ranks_col = mydb["ranks"]
        for item in ranks_col.find({"phonenumber": phoneNum}):
            return item['ranks']
    except Exception:
        return None

#保存结果进入mongodb
def save_result(phoneNum,movie_id,sc):

    result = {}
    result["phonenumber"] = phoneNum
    result["_id"] = phoneNum
    result["movies"] = []

    m_sdf = get_data.sprak_procuration_load_csv(params.data_file_root_path + os.sep + "movies.csv",sc)

    for id in movie_id:
        movie_dict = {}
        dict_one = m_sdf.filter("_c0="+id).collect()[0].asDict()
        movie_dict['geners'] = dict_one['_c2']
        movie_dict['title'] = dict_one['_c1']
        movie_dict['movie_id'] = dict_one['_c0']
        result["movies"].append(movie_dict)

    # movie_id为推荐的k个电影
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["movie"]
    result_col = mydb["result"]
    #使用insert
    if((result_col.find({"phonenumber": phoneNum}).count())==0):
        result_col.insert_one(result)
    else:
        #使用update
        result_col.update({"phonenumber": phoneNum},result)
    myclient.close()

#初始时随机选择
def select_random(sc,k,phoneNum):

    result = {}
    result["phonenumber"] = phoneNum
    result["_id"] = phoneNum
    result["movies"] = []

    m_sdf = get_data.sprak_procuration_load_csv(params.data_file_root_path + os.sep + "movies.csv", sc)\
        .rdd.takeSample(False,k)
    for item in m_sdf:
        movie_dict = {}
        dict_one = item.asDict()
        movie_dict['geners'] = dict_one['_c2']
        movie_dict['title'] = dict_one['_c1']
        movie_dict['movie_id'] = dict_one['_c0']
        result["movies"].append(movie_dict)

    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["movie"]
    result_col = mydb["result"]
    # 使用insert
    if ((result_col.find({"phonenumber": phoneNum}).count()) == 0):
        result_col.insert_one(result)
    else:
        # 使用update
        result_col.update({"phonenumber": phoneNum}, result)
    myclient.close()

if __name__ == '__main__':
    prec("19933055675",15)











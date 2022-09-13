#数据集路径
data_file_root_path = r"D:\codes\codes\moviemodel\data\ml-latest"
#模型权重存储路径
model_weight_file = r"D:\codes\codes\moviemodel\weights"
#模型tag
tag_en = "encoder"
tag_de = "decoder"
tag_ae = "autoencoder"

#训练轮次
epochs = 2000
#每轮次重复次数
verbose = 10
#学习率
lr = 0.0001
#模型
model_dir = r"D:\codes\codes\moviemodel\data"

movie_list_file_path = r"D:\codes\codes\moviemodel\data"

#输入向量大小
input_shape = 60000
#卷积核数量参数
latent_dim = 32
#训练样本批次大小
batch_size = 64

debug = True

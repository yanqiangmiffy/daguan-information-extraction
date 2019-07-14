# 设置lstm训练参数
class TrainingConfig(object):
    batch_size = 50
    # 学习速率
    lr = 1.0
    epoches = 1
    print_step = 5


class LSTMConfig(object):
    emb_size = 300  # 词向量的维数
    hidden_size = 128  # lstm隐向量的维数

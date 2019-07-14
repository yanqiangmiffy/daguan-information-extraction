import pickle
from gensim.models import Word2Vec


def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


# LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
# 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
def extend_maps(word2id, tag2id, for_crf=True):
    # word2id['<unk>'] = len(word2id)
    # word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
    if for_crf:
        # word2id['<start>'] = len(word2id)
        # word2id['<end>'] = len(word2id)
        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)

    return word2id, tag2id


def prepocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    assert len(word_lists) == len(tag_lists)

    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        if not test:  # 如果是测试数据，就不需要加end token了
            tag_lists[i].append("<end>")

    return word_lists, tag_lists


def train_format_transfer():
    f = open('../datagrand/train.txt', 'r', encoding='utf8')
    datas = f.readlines()
    f.close()
    f = open('normal_daguan_train.txt', 'w')
    all_words = set()
    all_labels = set()
    all_labels.add('o')
    for data in datas:
        words = []
        labels = []
        for texts in data.strip().split('  '):
            label = texts[-1]
            texts = texts[:-2].split('_')
            length = len(texts)
            if label == 'o':
                for i in range(length):
                    words.append(texts[i])
                    all_words.add(texts[i])
                    labels.append('o')
            else:
                if length == 1:
                    all_labels.add(label + '-S')
                    all_words.add(texts[0])
                    labels.append(label + '-S')
                    words.append(texts[0])
                elif length == 2:
                    all_labels.add(label + '-B')
                    all_labels.add(label + '-E')
                    all_words.add(texts[0])
                    all_words.add(texts[1])
                    labels.append(label + '-B')
                    labels.append(label + '-E')
                    words.append(texts[0])
                    words.append(texts[1])
                elif length > 2:
                    all_labels.add(label + '-B')
                    all_labels.add(label + '-M')
                    all_labels.add(label + '-E')
                    all_words.add(texts[0])
                    all_words.add(texts[-1])
                    labels.append(label + '-B')
                    words.append(texts[0])
                    for i in range(1, length - 1):
                        labels.append(label + '-M')
                        words.append(texts[i])
                        all_words.add(texts[i])
                    labels.append(label + '-E')
                    words.append(texts[-1])
        f.write(' '.join(words) + '|||' + ' '.join(labels) + '\n')
    word2id = {}
    label2id = {}
    for word in all_words:
        word2id[word] = len(word2id)
    for label in all_labels:
        label2id[label] = len(label2id)
    with open('word2id', 'wb') as f:
        pickle.dump(word2id, f)
    with open('tag2id', 'wb') as f:
        pickle.dump(label2id, f)
    print(word2id)
    print(label2id)
    f.close()


def test_format_transfer():
    f = open('../datagrand/test.txt', 'r', encoding='utf8')
    datas = f.readlines()
    f.close()
    f = open('normal_daguan_test.txt', 'w')
    for data in datas:
        words = []
        for texts in data.strip().split('  '):
            texts = texts.split('_')

            f.write(' '.join(texts) + '\n')
    f.close()


def corpus_format_transfer():
    f = open('../datagrand/corpus.txt', 'r', encoding='utf8')
    datas = f.readlines()
    f.close()
    f = open('daguan_languagemodel_corpus.txt', 'w')
    for data in datas:
        for texts in data.strip().split('  '):
            texts = texts.split('_')

            f.write(' '.join(texts) + '\n')
    f.close()


def submitFormat():
    """
    将预测出的标签转为提交结果
    :return:
    """
    f = open('next_dev_result/1.453_bilstmcrf_result.txt')
    labels = f.readlines()
    f.close()
    f = open('../datagrand/test.txt')
    texts = f.readlines()
    f.close()
    f = open('result.txt', 'w', encoding='utf8')
    for i, label in enumerate(labels):

        label = label.strip().split(' ')
        text = texts[i].strip().split('_')
        print(len(label), len(text))
        length = len(label)
        start = 0
        end = 0
        result = []
        print(i, label)
        while length > 0:
            if label[end] == 'o':
                while label[end] == 'o':
                    end += 1
                    length -= 1
                    if length == 0:
                        break
                result.append('_'.join(text[start:end]) + '/o')
                start = end
                if start == len(label):
                    break
            if 'S' in label[end]:
                # print(text[start:end] +'/'+label[end][0])
                result.append('_'.join(text[start:end + 1]) + '/' + label[end][0])
                end += 1
                length -= 1
                start = end
                if length == 0:
                    break
            if 'B' in label[end]:
                while 'E' not in label[end]:
                    end += 1
                    length -= 1
                    if length == 0:
                        break
                end += 1
                length -= 1
                result.append('_'.join(text[start:end]) + '/' + label[start][0])
                start = end
                if start == len(label):
                    break
            # if '' in label[end]:
        f.write('  '.join(result) + '\n')
    f.close()


# submitFormat()


def trainCharEmbedding():
    """
    训练词向量
    :return:
    """

    f = open('../datagrand/corpus.txt')
    datas = f.readlines()
    f.close()
    texts = []
    for data in datas:
        words = data.strip().split('_')
        texts.append(words)
    print(len(texts))
    model = Word2Vec(texts, size=300, iter=20, min_count=0, min_alpha=0, sg=1, hs=1, workers=64)
    model.save('charEmbedding_300dim')


# trainCharEmbedding()



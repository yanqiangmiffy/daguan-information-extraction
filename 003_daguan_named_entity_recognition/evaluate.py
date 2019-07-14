import time

from models.bilstm_crf import BILSTM_Model
from utils import save_model



def bilstm_train_and_eval(train_data, dev_data, test_data,
                          charEmbedding,word2id, tag2id, crf=True, remove_O=False):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data
    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = BILSTM_Model(vocab_size,charEmbedding, out_size, crf=crf)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)

    model_name = "bilstm_crf" if crf else "bilstm"
    save_model(bilstm_model, "./ckpts/"+str(bilstm_model.best_val_loss)[:5]+model_name+".pkl")

    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    print("评估{}模型中...".format(model_name))
    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)
    f = open('next_dev_result/'+str(bilstm_model.best_val_loss)[:5]+'_bilstmcrf_result.txt', 'w')
    for pred_tag_list in pred_tag_lists:
        f.write(' '.join(pred_tag_list) + '\n')
    f.close()



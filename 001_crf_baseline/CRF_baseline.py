import os
import numpy as np
from sklearn.model_selection import KFold


def micro_f1(sub_lines, ans_lines, split=' '):
    correct = []
    total_sub = 0
    total_ans = 0
    for sub_line, ans_line in zip(sub_lines, ans_lines):
        sub_line = set(str(sub_line).split(split))
        ans_line = set(str(ans_line).split(split))
        c = sum(1 for i in sub_line if i in ans_line) if sub_line != {''} else 0
        total_sub += len(sub_line) if sub_line != {''} else 0
        total_ans += len(ans_line) if ans_line != {''} else 0
        correct.append(c)
    p = np.sum(correct) / total_sub if total_sub != 0 else 0
    r = np.sum(correct) / total_ans if total_ans != 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) != 0 else 0
    print('total sub:', total_sub)
    print('total ans:', total_ans)
    print('correct: ', np.sum(correct), correct)
    print('precision: ', p)
    print('recall: ', r)
    print('f1', f1)


if __name__ == '__main__':
    os.system(
        "python 000.make_ner_format_data.py "
        "-org_file ../datagrand/test.txt "
        "-ner_file data/test_ner.txt "
        "-train test")

    train_numpy = []
    with open('../datagrand/train.txt') as files:
        for file in files:
            train_numpy.append(file)
    train_numpy = np.array(train_numpy)
    predict = []
    train = []
    kf = KFold(n_splits=5, shuffle=False, random_state=42)
    for i, (train_idx, test_idx) in enumerate(kf.split(train_numpy)):
        X_train = train_numpy[train_idx]
        X_test = train_numpy[test_idx]
        with open('data/train_cv_{}.txt'.format(i), 'w') as f:
            for t in X_train:
                f.write(t)
        with open('data/validate_cv_{}.txt'.format(i), 'w') as f:
            for t in X_test:
                f.write(t)

        os.system("python 000.make_ner_format_data.py "
                  "-org_file data/train_cv_{}.txt "
                  "-ner_file data/train_ner_cv_{}.txt "
                  "-train train".format(i, i))
        os.system("python 000.make_ner_format_data.py "
                  "-org_file data/validate_cv_{}.txt "
                  "-ner_file data/validate_ner_cv_{}.txt "
                  "-train train".format(i, i))

        print('train {}'.format(i))
        os.system("crf_learn "
                  "-f 3 "
                  "-c 1.5 "
                  "template "
                  "data/train_ner_cv_{}.txt  "
                  "models/train_model_{} ".format(i, i))
        #
        os.system("crf_test "
                  "-v2 "
                  "-m "
                  "models/train_model_{} "
                  "data/validate_ner_cv_{}.txt "
                  "> "
                  "submit/submit_validate_ner_cv_predict_prob_{}.txt".format(i, i, i))

        os.system("crf_test "
                  "-m "
                  "models/train_model_{} "
                  "data/validate_ner_cv_{}.txt "
                  "> "
                  "submit/submit_validate_ner_cv_predict_{}.txt".format(i, i, i))

        os.system("python 001.make_submit.py "
                  "-submit_file data/validate_cv_{}.txt "
                  "-predict_file submit/submit_validate_ner_cv_predict_{}.txt ".format(i, i)
                  )

        with open('data/validate_cv_{}.txt'.format(i)) as f:
            for ff in f:
                t = ff.split('  ')
                predict.append(t)

        with open('data/validate_cv_{}.txt'.format(i)) as f:
            for ff in f:
                t = ff.split('  ')
                train.append(t)

        print('predict {}'.format(i))
        os.system("crf_test "
                  "-v2 "
                  "-m "
                  "models/train_model_{} "
                  "data/test_ner.txt "
                  "> "
                  "submit/a_submit_test_ner_prob_{}.txt".format(i, i))
        os.system("crf_test "
                  "-m "
                  "models/train_model_{} "
                  "data/test_ner.txt "
                  "> "
                  "submit/a_submit_test_ner_predict_{}.txt".format(i, i))

    micro_f1(train, predict, split=' ')

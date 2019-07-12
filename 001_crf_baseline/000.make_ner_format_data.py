import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-org_file")
parser.add_argument("-ner_file")
parser.add_argument("-train")
args = parser.parse_args()

print('begin {}'.format(args.org_file))
with open(args.org_file) as f:
    with open(args.ner_file, 'w') as fw:
        for i,sentence in enumerate(f):
            delimiter = '\t'
            words = sentence.replace('\n','').split('  ')
            for j,word in enumerate(words):
                split_word = word.split('/')
                if args.train == 'train':
                    tag = split_word[1]
                else:
                    tag = 'O'
                word_meta = split_word[0]
                word_meta_split = word_meta.split('_')
                meta_len = len(word_meta_split)
                if tag == 'a':
                    if meta_len == 1:
                        fw.write(word_meta_split[0] + delimiter + 'B_a' + '\n')
                        # fw.write(word_meta_split[0] + delimiter + 'W_a' + '\n')
                        # fw.write(word_meta_split[0] + delimiter + 'O' + '\n')
                    else:
                        for k, char in enumerate(word_meta_split):
                            if k == 0:
                                fw.write(char + delimiter + 'B_a' + '\n')
                            elif k == meta_len - 1:
                                # fw.write(char + delimiter + 'E_a' + '\n')
                                fw.write(char + delimiter + 'I_a' + '\n')
                            else:
                                # fw.write(char + delimiter + 'M_a' + '\n')
                                fw.write(char + delimiter + 'I_a' + '\n')
                elif tag == 'b':
                    if meta_len == 1:
                        fw.write(word_meta_split[0] + delimiter + 'B_b' + '\n')
                        # fw.write(word_meta_split[0] + delimiter + 'W_b' + '\n')
                        # fw.write(word_meta_split[0] + delimiter + 'O' + '\n')
                    else:
                        for k, char in enumerate(word_meta_split):
                            if k == 0:
                                fw.write(char + delimiter + 'B_b' + '\n')
                            elif k == meta_len - 1:
                                fw.write(char + delimiter + 'I_b' + '\n')
                            else:
                                # fw.write(char + delimiter + 'M_b' + '\n')
                                fw.write(char + delimiter + 'I_b' + '\n')
                elif tag == 'c':
                    if meta_len == 1:
                        fw.write(word_meta_split[0] + delimiter + 'B_c' + '\n')
                        # fw.write(word_meta_split[0] + delimiter + 'W_c' + '\n')
                        # fw.write(word_meta_split[0] + delimiter + 'O' + '\n')
                    else:
                        for k, char in enumerate(word_meta_split):
                            if k == 0:
                                fw.write(char + delimiter + 'B_c' + '\n')
                            elif k == meta_len - 1:
                                fw.write(char + delimiter + 'I_c' + '\n')
                            else:
                                # fw.write(char + delimiter + 'M_c' + '\n')
                                fw.write(char + delimiter + 'I_c' + '\n')
                else:
                    if meta_len == 1:
                        fw.write(word_meta_split[0] + delimiter + 'O' + '\n')
                    else:
                        for k, char in enumerate(word_meta_split):
                            fw.write(char + delimiter + 'O' + '\n')
            fw.write('\n')
print('finish {}'.format(args.ner_file))
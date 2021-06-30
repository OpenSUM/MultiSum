import re
import time

import numpy as np

from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")



f1 = open('reviews_Home_and_Kitchen_token.txt', 'w',encoding='utf-8')
filename = open('reviews_Home_and_Kitchen.txt','r',encoding='utf-8')
lines = filename.readlines()
filename.close()
count = 0

for i in range(0, len(lines), 3):
    if count % 1000 == 0:
        print('finish: ', count)

    flag = False
    right = []
    sentence1 = tokenizer.tokenize(lines[i].replace('\n',''))
    if len(sentence1) < 16 or len(sentence1) > 800:
        continue
    else:
        right.append(sentence1)
    sentence2 = lines[i + 1].replace('\n','')
    right.append(sentence2)
    sentence3 = tokenizer.tokenize(lines[i + 2].replace('\n',''))
    if len(sentence3) < 4:
        continue
    else:
        right.append(sentence3)

    for sentence in right:
        #print(sentence)
        f1.write((' '.join(sentence))+'\n')
        flag = True

    if flag:
        count = count + 1

f1.close()


print('total: ', count)

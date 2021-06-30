import re
import time

import numpy as np
import json

import os


f1 = open('reviews_Home_and_Kitchen.txt', 'w')

filename = open('reviews_Home_and_Kitchen.json','r')
lines = filename.readlines()
filename.close()
count = 0
for i in lines:
    if (count%1000==0):
        print('finish: ', count)
    data=json.loads(i)
    article = data["reviewText"]
    summary = data["summary"]
    number = data["overall"]
    f1.write(article.lower()+'\n')
    f1.write(str(number)+'\n')
    f1.write(summary.lower()+'\n')
    count = count + 1
f1.close()


print('total: ', count)

import random

f = open('reviews_Home_and_Kitchen_token.txt', 'r')
lines = f.readlines()
f.close()

lists = list()

for i in range(0, len(lines), 3):
    my_list = list()
    my_list.append(lines[i])
    my_list.append(lines[i + 1])
    my_list.append(lines[i + 2])
    lists.append(my_list)

random.shuffle(lists)

f1 = open('valid.src', 'w')
f2 = open('valid.dst', 'w')
f3 = open('valid.senti', 'w')

f4 = open('test.src', 'w')
f5 = open('test.dst', 'w')
f6 = open('test.senti', 'w')

f7 = open('train.src', 'w')
f8 = open('train.dst', 'w')
f9 = open('train.senti', 'w')

for i in range(0, 9000):
    f1.write(lists[i][0])
    f2.write(lists[i][2])
    f3.write(lists[i][1])

f1.close()
f2.close()
f3.close()

for i in range(9000, 18000):
    f4.write(lists[i][0])
    f5.write(lists[i][2])
    f6.write(lists[i][1])

f4.close()
f5.close()
f6.close()

for i in range(18000, len(lists)):
    f7.write(lists[i][0])
    f8.write(lists[i][2])
    f9.write(lists[i][1])

f7.close()
f8.close()
f9.close()

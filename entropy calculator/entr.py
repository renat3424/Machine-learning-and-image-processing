import math as m
from PIL import Image
import matplotlib.pyplot as plt
import collections
def entropy(dict):
    S=0
    for i in dict.keys():
        S+=dict[i]*m.log(1/dict[i], 2)
    return S

with open("neznajka-na-lune-5806850.txt", "r") as f:
    data=f.read()
    data=data.lower()
    dict={}
    num=0
    print(len(data))
    for i in set(data):
        if i not in ["\n", "\t", "\xa0"]:
            dict[i]=data.count(i)
            num+=data.count(i)
    for i, k in list(dict.items()):
        if k in range(0, 10):
            num-=k
            dict.pop(i)
    for i, k in dict.items():
        dict[i]=k/num
    print(dict)
    plt.figure(0, figsize=(10, 4))

    hist = plt.bar(dict.keys(), dict.values())
    plt.xlabel('алфавит')
    plt.ylabel('частоты символов')


    plt.show()

    print(entropy(dict))
    print(entropy(dict)*len(data))

with Image.open("neznajka-na-lune-5806850-237x379.jpg").convert("L") as im:
     im.show()
     width,height=im.size
     print(width,height)
     dict={}
     for i in im.getdata():
         if i in dict.keys():
            dict[i]+=1
         else:
            dict[i]=1


     for i in dict.keys():
         dict[i]/=width*height
     od=collections.OrderedDict(sorted(dict.items()))
     plt.figure(0, figsize=(10, 4))

     plt.plot(od.keys(), od.values())
     print(od)
     plt.xlabel('интенсивности пикселей')
     plt.ylabel('частоты')

     plt.show()

     print(entropy(dict))
     print(width*height*entropy(dict))
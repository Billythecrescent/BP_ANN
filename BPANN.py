#BPANN.py
from sys import argv
from random import randint
from math import *

######## Illustration ########
# Two-layer BP neural network
# input layer node: 3 (Three attributes)
# output layer node: 1 (1 or 0)
# intermediate layer node: uncertain
# activation function: sigmoid function
# 
######## END ########


#1. Read the titanic.dat file
def readDat(filename):
    f = open(filename,'r')
    lines = f.readlines()
    pclass,age,sex = [],[],[]
    people = []
    for i in lines:
        if i[0] != '@':
            values = i.split(',')
            pclass.append(float(values[0]))
            age.append(float(values[1]))
            sex.append(float(values[2]))
            attributes = map(float,values[:-1])
            label = float(values[-1][:-1]) #remove the \n
            person = {"attribute":attributes,"label":label}
            people.append(person)
    return people,pclass,age,sex
    #print(people)
    #print(len(people))

#readDat("titanic.dat")

#2. Standardise the data using Range method, to [0,1]
def ReadplusStandard(filename):
    people,pclass,age,sex = readDat(filename)
    maxclass,minclass = max(pclass),min(pclass)
    maxage,minage = max(age),min(age)
    maxsex,minsex = max(sex),min(sex)
    standclass = lambda x: (x-minclass)/(maxclass-minclass)
    standage = lambda x: (x-minage)/(maxage-minage)
    standsex = lambda x: (x-minsex)/(maxsex-minsex)
    for person in people:
        a,b,c = person["attribute"]
        person["attribute"] = standclass(a),standage(b),standsex(c)
    return people
    #print(people)

#ReadplusStandard("titanic.dat")

#3. Divide the data into two sets, with the ratio of 7:3
#people: the standardlised data structure
#ratio: ratio of training data and test data, from 0 to 1
def DataDevide(people,ratio):
    randomintlist = []
    traindata = []
    testdata = []
    for i in range(int(floor(len(people)*ratio))):
        index = randint(0,len(people)-1)
        while index in randomintlist:
            index = randint(0,len(people)-1)
        randomintlist.append(index)
        traindata.append(people[index])
    for i in range(len(people)-1):
        if i not in randomintlist:
            testdata.append(people[i])
    return traindata,testdata
    #print(len(traindata))
    #print(len(testdata))

#4. BP neural network classification
#input: traindata; intermediate layer unit number, k
#output:weight list
def BPNetwork(traindata,k):
    #初始化各个权值
    ##节点个数
    input_node = 3
    output_node = 1
    inter_node = k
    ##初始化权重为0
    weight_num = input_node * output_node * inter_node + inter_node + output_node #权重个数
    weight = [[[0]*(input_node+1)]*(inter_node),[[0]*(inter_node+1)]*(output_node)]
    #[ 0 0 0 0*               ]
    #[ 0 0 0 0* ,  0 0 0 0 0* ]
    #[ 0 0 0 0*               ]
    #[ 0 0 0 0*               ]   #每个矩阵最后一行是阈值节点的权值
    
    #输入训练集数据，开始训练
    for i in traindata:
        inputdata = i["attribute"]+[1]
        

#BPANN.py
from sys import argv
from random import randint
from math import *
import numpy as np

######## Illustration ########
# Two-layer BP neural network
# input layer node: 3 (Three attributes)
# output layer node: 1 (1 or 0)
# intermediate layer node: uncertain
# activation function: sigmoid function
# 
######## END ########


def sigmoid(x):
    '''
    sigmoid function: activate function
    input: float
    output: float
    '''
    return 1.0 / (1.0 + np.exp(-x))


'''
def sigmoid(x):
    return (.5 * (1 + np.tanh(.5 * x)))
'''

'''def sigmoid(value):
    if -value > np.log(np.finfo(type(value)).max):
        return 0.0    
    a = exp(-value)
    return 1.0/ (1.0 + a)
'''
def sigmod_derivate(x):
    '''
    sigmoid_derivate function
    the level 1 derivation of sigmoid function
    input: unactivated numebr, float
    '''
    return sigmoid(x)*(1-sigmoid(x))

#1. Read the titanic.dat file
def readDat(filename):
    '''
    read .dat file to get the data set
    input: filepath
    output: people,pclass,age,sex
    people: [diary{“attribute”:list[3]”,”label”:int}, diary{“attribute”:list[3]”,”label”:int}, …]
    pclass: the extracted class attributes value list
    age: the extracted age attributes value list
    sex: the extracted sex attributes value list
    '''
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
            attributes = list(map(float,values[:-1]))
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
        d = person["label"]
        person["attribute"] = [standclass(a),standage(b),standsex(c)]
        if d == -1.0:
            person["label"] = 0.0
    #print(people)
    return people
    

#ReadplusStandard("titanic.dat")

#3. Divide the data into two sets, with the ratio of 7:3
#people: the standardlised data structure
#ratio: ratio of training data and test data, from 0 to 1
def DataDevide(people,ratio=0.7):
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

def Forward(inputdata,weight,bias,k):
    '''
    Forward propagation algorithm
    input: inputdata, weight, bias, k
    inputdata: the 3-element tuple of the attributes, (class, age, sex)
    weight: the weight list containing all the weight of the BP neural network
    bias: the bias list containing all the bias of the BP neural network
    k: the number of neuron in hidden layer

    This function is used for the classification of the testdata
    '''
    #进行正向传播
    L1array = inputdata*(weight[0]) + bias[0] #output: 1*k
    actiL1array = []
    for i in range(k):
        actiL1array.append(sigmoid(L1array[0,i]))
    actiL1array = np.mat(actiL1array) #output: 1*4
    L2value = float(actiL1array*(weight[1]) + bias[1])  #由此得到了正向传播的输出值
    actiL2value = sigmoid(L2value) #float
    return actiL2value

def ForwardABackward(inputdata,label,weight,bias,k,LearnRatio):
    '''
    Forward propagation function and Back propagation function together
    input: inputdata, label, weight, bias, k, LeanRatio
    inputdata: the 3-element tuple of the attributes, (class, age, sex)
    label: the label of the train data, used to calculate the error
    weight: the weight list containing all the weight of the BP neural network
    bias: the bias list containing all the bias of the BP neural network
    k: the number of neuron in hidden layer
    LearnRatio: the learn ratio of the neural network

    The funtion is used to update the weight list and bias list
    '''
    #进行正向传播
    L1array = inputdata*(weight[0]) + bias[0] #output: 1*k
    actiL1array = []
    for i in range(k):
        actiL1array.append(sigmoid(L1array[0,i]))
    actiL1array = np.mat(actiL1array) #output: 1*4
    L2value = float(actiL1array*(weight[1]) + bias[1])  #由此得到了正向传播的输出值
    actiL2value = sigmoid(L2value) #float

    #进行反向传播
    ##计算误差
    change_b = [np.zeros(b.shape) for b in bias]
    change_w = [np.zeros(w.shape) for w in weight]
    L2error = actiL2value-label #float
    L2Delta = L2error*sigmod_derivate(L2value) #float
    
    ##计算第二层权值变化
    change_b[-1] = L2Delta
    # 乘于前一层的输出值
    change_w[-1] = (actiL1array.T)*L2Delta #output: 4*1
    
    ##计算第一层权值变化
    sigmoid_prime = np.multiply(actiL1array,(1-actiL1array))
    L1Delta = np.multiply(L2Delta*(weight[1].T),sigmoid_prime) #output: 1*4
    change_b[-2] = L1Delta
    change_w[-2] = (np.mat(inputdata).T)*L1Delta

    '''
    ##更新第二层权值
    weight[1] -= LearnRatio * change_w[-1]
    bias[1] -= LearnRatio * change_b[-1] 
    ##更新第一层权值
    weight[0] -= LearnRatio * change_w[-2]
    bias[0] -= LearnRatio * change_b[-2]
    '''
    return change_w,change_b

def GD(traindata,k,LearnRatio,num):
    '''
    Gradient Descent algorithm
    '''
    attrinum = len(traindata[0]["attribute"])
    bias = [np.mat(np.random.normal(size=(1,y)),dtype='float64') for y in [k,1] ]
    weight = [np.mat(np.random.normal(size=(x,y)),dtype='float64') for x,y in [(attrinum,k),(k,1)] ]
    #[ 0 0 0 0     0  ]
    #[ 0 0 0 0  ,  0  ]
    #[ 0 0 0 0     0  ]
    #[             0  ]   #每个矩阵最后一行是阈值节点的权值
    #print(weight)
    #print(bias)
    
    for j in range(num):
        #初始化累加权值和偏移量
        nabla_b = [np.zeros(b.shape) for b in bias]
        nabla_w = [np.zeros(w.shape) for w in weight]
        for i in traindata:
            inputdata,label = i["attribute"],i["label"]
            #进行正向传播和反向传播
            change_w,change_b = ForwardABackward(inputdata,label,weight,bias,k,LearnRatio)
            #累加权值偏差
            nabla_b[0] += change_b[-2]
            nabla_b[1] += change_b[-1]
            nabla_w[0] += change_w[-2]
            nabla_w[1] += change_w[-1] 
        #更新权值
        datalens = len(traindata)
        weight[0] -= LearnRatio/datalens*nabla_w[0]
        weight[1] -= LearnRatio/datalens*nabla_w[1]
        bias[0] -= LearnRatio/datalens*nabla_b[0]
        bias[1] -= LearnRatio/datalens*nabla_b[1]

    return weight,bias

def SGD(traindata,k,LearnRatio,num):
    '''
    Gradient Descent algorithm
    '''
    attrinum = len(traindata[0]["attribute"])
    bias = [np.mat(np.random.normal(size=(1,y)),dtype='float64') for y in [k,1] ]
    weight = [np.mat(np.random.normal(size=(x,y)),dtype='float64') for x,y in [(attrinum,k),(k,1)] ]
    #[ 0 0 0 0     0  ]
    #[ 0 0 0 0  ,  0  ]
    #[ 0 0 0 0     0  ]
    #[             0  ]   #每个矩阵最后一行是阈值节点的权值
    #print(weight)
    #print(bias)
    
    for j in range(num):
        #初始化累加权值和偏移量
        nabla_b = [np.zeros(b.shape) for b in bias]
        nabla_w = [np.zeros(w.shape) for w in weight]
        index = randint(0,len(traindata)-1)
        i = traindata[index]
        inputdata,label = i["attribute"],i["label"]
        #进行正向传播和反向传播
        change_w,change_b = ForwardABackward(inputdata,label,weight,bias,k,LearnRatio)
        
        #更新权值
        
        weight[1] -= LearnRatio * change_w[-1]
        bias[1] -= LearnRatio * change_b[-1] 
        ##更新第一层权值
        weight[0] -= LearnRatio * change_w[-2]
        bias[0] -= LearnRatio * change_b[-2]
        #weight = [w-(LearnRatio)*nw for w, nw in zip(weight, nabla_w)]
        #bias = [b-(LearnRatio)*nb for b, nb in zip(bias, nabla_b)]

    return weight,bias


def miniSGD(traindata,k,LearnRatio,num,minibatch):
    '''
    Stochastic Gradient Descent algorithm
    '''
    attrinum = len(traindata[0]["attribute"])
    bias = [np.mat(np.random.normal(size=(1,y)),dtype='float64') for y in [k,1] ]
    weight = [np.mat(np.random.normal(size=(x,y)),dtype='float64') for x,y in [(attrinum,k),(k,1)] ]
    #[ 0 0 0 0     0  ]
    #[ 0 0 0 0  ,  0  ]
    #[ 0 0 0 0     0  ]
    #[             0  ]   #每个矩阵最后一行是阈值节点的权值
    #print(weight)
    #print(bias)
    
    #利用小样本集进行训练
    for j in range(num):
        nabla_b = [np.zeros(b.shape) for b in bias]
        nabla_w = [np.zeros(w.shape) for w in weight]

        #划分小样本集
        randomlist = []
        for i in range(minibatch):
            index = randint(0,len(traindata)-1)
            while index in randomlist:
                index = randint(0,len(traindata)-1)
            randomlist.append(index)
    
        for i in randomlist:
            person = traindata[i]
            inputdata,label = person["attribute"],person["label"]
            #进行正向传播和反向传播
            change_w,change_b = ForwardABackward(inputdata,label,weight,bias,k,LearnRatio)
            #累加权值偏差
            #nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, change_b)]
            #nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, change_w)]
            nabla_b[0] += change_b[-2]
            nabla_b[1] += change_b[-1]
            nabla_w[0] += change_w[-2]
            nabla_w[1] += change_w[-1] 
        #更新权值
        datalens = minibatch
        weight[0] -= LearnRatio/datalens*nabla_w[0]
        weight[1] -= LearnRatio/datalens*nabla_w[1]
        bias[0] -= LearnRatio/datalens*nabla_b[0]
        bias[1] -= LearnRatio/datalens*nabla_b[1]
        #weight = [w-(LearnRatio/datalens)*nw for w, nw in zip(weight, nabla_w)]
        #bias = [b-(LearnRatio/datalens)*nb for b, nb in zip(bias, nabla_b)]
        
    return weight,bias

#4. BP neural network classification
def BPNetwork(traindata,k,LearnRatio,method,num):
    '''
    BPNetwork
    input: 
    traindata; 
    intermediate layer unit number, k; 
    LearnRatio;
    method: GD or SGD
    output: updated weight, updated bias
    '''
    #初始化各个权值
    ##节点个数
    attrinum = len(traindata[0]["attribute"])
    input_node = 3
    output_node = 1
    if method == "GD":
        weight,bias = GD(traindata,k,LearnRatio,num)
    elif method == "SGD":
        weight,bias = SGD(traindata,k,LearnRatio,num)
    elif method == "miniSGD":
        minibatch = 10
        weight,bias = miniSGD(traindata,k,LearnRatio,num,minibatch)
    else:
        print("method error, please check your method.\n")
    return weight,bias

#5. Test the accuracy of BP network
def TestAccurary(traindata,testdata,k,LearnRatio,method,num):
    weight,bias = BPNetwork(traindata,k,LearnRatio,method,num)
    #print(weight)
    #print(bias)
    Truenum = 0
    for i in testdata:
        output = 0
        inputdata,label = i["attribute"],i["label"]
        output = Forward(inputdata,weight,bias,k)
        #print(output,label)
        if output >= 0.5:
            output = 1.0
        else:
            output = 0.0
        if output == label:
            Truenum+=1
    return Truenum/len(testdata)
    

'''
def main(argv):
    
    if len(argv) == 1:
        print("BPANN.py: A BP neural network algorithm program to predict the survival of a person in the titanic incident.\n")
        print("This neural network is only 3 layers, with input layer, output layer and only one hidden layer.\n")
        print("Usage:\npython BPANN.py filepath k method learn_ratio train_round\n")
        print("filepath\n\tthe full path of the .dat or .csv file, relative or absolute all both accepted.\n")
        print("k\n\tthe number of hidden layer, which has to be set by the user.\n")
        print("method\n\tEither 'SGD' or 'GD', to be Stochastic Gradient Descent method, or Gradient Descent method. \n")
        print("lean_ratio\n\tThe lean ratio of the BP network.\n")
        print("train_round\n\tThe number of rounds the training process takes.\n")
    elif len(argv) > 1:
        filename = argv[1]
        if len(argv) == 2:
            k = 4
            method = "GD"
            LearnRatio = 0.5
            num = 50
        elif len(argv) == 3:
            k = int(argv[2])
            method = "GD"
            LearnRatio = 0.5
            num = 50
        elif len(argv) == 4:
            k = int(argv[2])
            method = argv[3]
            LearnRatio = 0.5
            num = 50
        elif len(argv) == 5:
            k = int(argv[2])
            method = argv[3]
            LearnRatio = float(argv[4])
            num = 50
        else:
            k = int(argv[2])
            method = argv[3]
            LearnRatio = float(argv[4])
            num = int(argv[5])
    
        people = ReadplusStandard("titanic.dat")
        traindata,testdata = DataDevide(people)
        #k = 4
        #LearnRatio = 0.1
        #method = "SGD"
        #num = 100

        TrueValue = TestAccurary(traindata,testdata,k,LearnRatio,method,num)
        print("Truevalue: ",TrueValue)
'''

def main():
    people = ReadplusStandard("titanic.dat")
    traindata,testdata = DataDevide(people)
    k = 4
    LearnRatio = 0.1
    method = "miniSGD"
    num = 10000

    TrueValue = TestAccurary(traindata,testdata,k,LearnRatio,method,num)
    print("Truevalue: ",TrueValue)

main()
#main(argv)

'''people = ReadplusStandard("titanic.dat")
traindata,testdata = DataDevide(people)
weight_num = 3 * 1 * 4 + 4 + 1 #权重个数
L1weight = np.mat(np.random.normal(size=(4,3+1)))  #正态分布随机初始化第一层权值(中间节点行，属性个数+1列)
L2weight = np.mat(np.random.normal(size=(1,4))) #正态分布初始化第二层权值(一行，中间节点列)
weight = [L1weight,L2weight]
print(weight)
for i in range(10):
    #print(type(traindata[0]["attribute"]))
    weight = ForwardABackward(traindata[0]["attribute"]+[1],traindata[0]["label"],weight,4,0.2)
    print(weight)
'''
'''
people = ReadplusStandard("titanic.dat")
traindata,testdata = DataDevide(people)
k = 4
LearnRatio = 0.1
method = "GD"
BPNetwork(traindata,k,LearnRatio,method,10)
'''
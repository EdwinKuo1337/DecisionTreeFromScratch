## you cannot use sklearn
import csv
import glob
import numpy as np
import csv
from Function import *
import operator
import math
import matplotlib.pyplot as plt

def read_csv(filename):
    t = []
    csv_file = open(filename,'r')
    i =0
    for row in csv.reader(csv_file):
        # if (i > 2):
        t.append(row)
        # i+=1
    # print (t)
    t = np.array(t,dtype = np.float32)
    return t

def accF (x,y):
    acc = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            acc +=1
    acc = acc / len(x)
    return acc 

def feature_ex(x):
    t = []
    x = np.array(x)
    indexAx = 0
    indexAy = 1
    indexAz = 2
    indexGx = 3
    indexGy = 4
    indexGz = 5
    indexTotalAcc = 6
    indexTotalGyro = 7
    indexRoll = 8
    indexPitch = 9
    
    totalAcc = getTotalAxes(x[indexAx],x[indexAy],x[indexAz])
    totalGyro = getTotalAxes(x[indexGx],x[indexGy],x[indexGz])
    roll = getRoll(x[indexAx],x[indexAz])
    pitch = getPitch(x[indexAy],x[indexAz])

    processedKoalaData = np.ones((10,40))

    for i in range(6):
        processedKoalaData[i] = copy.deepcopy(x[i])
    processedKoalaData[6] = copy.deepcopy(totalAcc)
    processedKoalaData[7] = copy.deepcopy(totalGyro)
    processedKoalaData[8] = copy.deepcopy(roll)
    processedKoalaData[9] = copy.deepcopy(pitch)



    mean = getMean2D(processedKoalaData)
    t.append(mean)


    return t





##################################
## load data
##################################

sensor = []
label = []
feature = []
f = glob.glob(r'40_data/down'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([0])

f = glob.glob(r'40_data/up'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([0])

f = glob.glob(r'40_data/left'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([1])
f = glob.glob(r'40_data/right'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([1])

f = glob.glob(r'40_data/CW'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([2])

f = glob.glob(r'40_data/CCW'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([3])



f = glob.glob(r'40_data/VLR'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([4])

f = glob.glob(r'40_data/VRL'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([5])


f = glob.glob(r'40_data/non'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([6])


sensor = np.array(sensor)
# print ('sensor shape is :',sensor.shape)

feature_name = [ "x["+str(i)+"]" for i in range((sensor.shape[1]*sensor.shape[2]))]
sensor = np.reshape(sensor,(sensor.shape[0],sensor.shape[1]*sensor.shape[2]))
label  = np.array(label)
# print ('sensor shape after is :',sensor.shape)
# print ('label shape is :',label.shape)
# print ('label is :',label)
label = np.reshape(label, (-1, 1))
newData = np.concatenate((sensor,label),axis=1)
# split train-test
# np.random.shuffle(newData)
train_X, val_X, train_y, val_y = newData[49:851,:], newData[0:int(0.8*900),-1], newData[int(0.8*900)+1:900,:], newData[int(0.8*900)+1:900,-1]
# train_X, val_X, train_y, val_y = newData[0:int(0.8*900),:], newData[0:int(0.8*900),-1], newData[int(0.8*900)+1:900,:], newData[int(0.8*900)+1:900,-1]
# val_X = [int(item) for item in val_X]
# val_y = [int(item) for item in val_y]

####################################
# # build decision tree
def class_counts(data):
    label = [0,0,0,0,0,0,0]
    for row in data:
        label[int(row[-1])] = label[int(row[-1])] + 1
    return label

class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def ask(self, example):
        return example[self.column] <= self.value

def partition(data, question):
    true_data, false_data = [], []
    for row in data:
        if question.ask(row):
            true_data.append(row)
        else:
            false_data.append(row)
    return true_data, false_data

def gini(data):
    counts = class_counts(data)
    sumOfProb = 0
    if len(data) == 0:
        return sumOfProb
    # for i in range(len(counts)):
    #     probOfLabel = float(counts[i]) / len(data)
    #     sumOfProb = sumOfProb + probOfLabel*probOfLabel
    # GiniImpurity = 1 - sumOfProb
    for i in range(len(counts)):
        probOfLabel = float(counts[i]) / len(data)
        if probOfLabel != 0:
            sumOfProb = sumOfProb - probOfLabel * math.log(probOfLabel, 2)
    
    return sumOfProb

def info_gain(left, right, uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return uncertainty - p * gini(left) - (1 - p) * gini(right)

def splitQuestion(data):
    bestPerformance = 0
    questionToAsk = None
    uncertainty = gini(data)
    features = len(data[0]) - 1
    for column in range(features):
        values = []
        for i in range(len(data)):
            values.append(data[i][column])
        
        for value in values:
            question = Question(column, value)
            true_data, false_data = partition(data, question)

            gain = info_gain(true_data, false_data, uncertainty)
            if gain > bestPerformance:
                bestPerformance, questionToAsk = gain, question

    return bestPerformance, questionToAsk

class Leaf:
    def __init__(self, data):
        allClass = class_counts(data)
        self.predictions = np.argmax(allClass)

class TreeNode:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(data, maxDepth, currentDepth):
    gain, question = splitQuestion(data)
    if gain == 0 or currentDepth == maxDepth:
        return Leaf(data)

    true_data, false_data = partition(data, question)

    true_branch = build_tree(true_data, maxDepth, currentDepth + 1)
    false_branch = build_tree(false_data, maxDepth, currentDepth + 1)

    return TreeNode(question, true_branch, false_branch)

def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.ask(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)   

def printTree(Tree):
    print(1)
    printTree(Tree.true_branch)
    printTree(Tree.false_branch)
####################################

# # predict training data or testing data

bestAccuracy = 0
bestTree = None

# for i in range(50):
#     np.random.shuffle(newData)
#     train_X, val_X, train_y, val_y = newData[0:int(0.8*900),:], newData[0:int(0.8*900),-1], newData[int(0.8*900)+1:900,:], newData[int(0.8*900)+1:900,-1]
#     val_X = [int(item) for item in val_X]
#     val_y = [int(item) for item in val_y]

#     decisionTree = build_tree(train_X, 6, 0)
#     val_y_predicted = []
#     for row in train_y:
#         val_y_predicted.append(classify(row, decisionTree))
#     acc = accF(val_y_predicted,val_y)
#     # print ('val_y acc is :',acc)
#     if acc > bestAccuracy:
#         bestTree = decisionTree
#         bestAccuracy = acc
#         print("bestAccuracy = ", str(bestAccuracy))
#         print("update model!!! Depth = 6, i = ", i)

# for i in range(50):
#     np.random.shuffle(newData)
#     train_X, val_X, train_y, val_y = newData[0:int(0.8*900),:], newData[0:int(0.8*900),-1], newData[int(0.8*900)+1:900,:], newData[int(0.8*900)+1:900,-1]
#     val_X = [int(item) for item in val_X]
#     val_y = [int(item) for item in val_y]

#     decisionTree = build_tree(train_X, 7, 0)
#     val_y_predicted = []
#     for row in train_y:
#         val_y_predicted.append(classify(row, decisionTree))
#     acc = accF(val_y_predicted,val_y)
#     # print ('val_y acc is :',acc)
#     if acc > bestAccuracy:
#         bestTree = decisionTree
#         bestAccuracy = acc
#         print("bestAccuracy = ", str(bestAccuracy))
#         print("update model!!! Depth = 7, i = ", i)

# for i in range(50):
#     np.random.shuffle(newData)
#     train_X, val_X, train_y, val_y = newData[0:int(0.91*900),:], newData[0:int(0.91*900),-1], newData[int(0.91*900)+1:900,:], newData[int(0.91*900)+1:900,-1]
#     val_X = [int(item) for item in val_X]
#     val_y = [int(item) for item in val_y]

#     decisionTree = build_tree(train_X, 3, 0)
#     val_y_predicted = []
#     for row in train_y:
#         val_y_predicted.append(classify(row, decisionTree))
#     acc = accF(val_y_predicted,val_y)
#     # print ('val_y acc is :',acc)
#     if acc > bestAccuracy:
#         bestTree = decisionTree
#         bestAccuracy = acc
#         print("bestAccuracy = ", str(bestAccuracy))
#         print("update model!!! Depth = 3, i = ", i)

# np.random.shuffle(newData)

# train_X = newData
# val_X = [int(item) for item in val_X]
# val_y = [int(item) for item in val_y]

bestTree = build_tree(train_X, 6, 0)
# printTree(bestTree)
# exit()
# val_y_predicted = []
# for row in train_y:
#     val_y_predicted.append(classify(row, decisionTree))

subjects = []
for i in range(0,71):
    if len(str(i+1)) == 2:
        subjects.append(str(i+1))
    else:
        subjects.append("0" + str(i+1))

test = []
for subject in subjects:
    f = glob.glob(r'testdata/'+subject+'.csv')
    for i in range(len(f)):
        t = read_csv(f[i])
        if (len(t[0])) == 40:
            t = feature_ex(t)
            test.append(t)
test = np.array(test)
test = np.reshape(test,(test.shape[0],test.shape[1]*test.shape[2]))

ans = [["Id","Category"]]
i = 0
for row in test:
    ans.append([str(subjects[i]) + ".csv",classify(row, bestTree)])
    i = i + 1
with open('ans.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(ans)



exit()

# # # label is 
print(val_y)

acc = accF(val_y_predicted,val_y)
print ('val_y acc is :',acc)
# acc = accF(train_X_predicted,train_y)
# print ('train_y acc is :',acc)



### export your testing prediction to .csv
### upload to Kaggle and get your score 


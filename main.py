import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from BackPropagation import NeuralNet
import time

data = pd.read_csv('Data.csv')
classLabel = pd.read_csv('Class.csv')

for i in range(1, 6):
    print('Train: ', (10 - i) * 10, '%')
    print('Test: ', i * 10, '%')
    trainData, testData, trainClass, testClass = train_test_split(data, classLabel, test_size=i/10)

    model = NeuralNet(3, 0.1)
    startTime = time.time()
    model.train(trainData, trainClass)
    endTime = time.time()
    predictedClass = model.test(testData)

    print('Time: ', (endTime - startTime))
    print('Accuracy: ', round(accuracy_score(testClass, predictedClass) * 100, 2), '%')
    print('=========================================')

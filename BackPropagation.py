import random
import math
import numpy as np

HiddenLayerNode = []
OutputLayerNode = []


class NeuralNet:

    def __init__(self, nHiddenNode, errorTolerance):
        self.nHiddenNode = nHiddenNode
        self.errorTolerance = errorTolerance

    def train(self, trainData, trainClass):
        global HiddenLayerNode, OutputLayerNode

        trainData = np.array(trainData)
        trainClass = np.array(trainClass)

        nFeature = trainData.shape[1]
        nData = trainData.shape[0]
        self.nOutputNode = math.ceil(math.log(len(np.unique(trainClass)), 2))

        HiddenLayerNode = [HiddenNode(nFeature, 0.4, 0.4) for _ in range(self.nHiddenNode)]
        OutputLayerNode = [OutputNode(self.nHiddenNode, 0.6, 0.6) for _ in range(self.nOutputNode)]

        index = 0
        while index < nData:
            # TODO: send data to hidden layer nodes
            for node in HiddenLayerNode:
                node.sendData(trainData[index])

            # TODO: send data from hidden layer to output layer
            classLabel = (self.nOutputNode - len(bin(trainClass[index, 0])[2:])) * '0' + bin(trainClass[index, 0])[2:]
            classLabel = classLabel[::-1]
            # calculate the total error
            error = 0
            for i in range(self.nOutputNode):
                OutputLayerNode[i].sendData([n.output for n in HiddenLayerNode])
                error += OutputLayerNode[i].getError(int(classLabel[i])) ** 2
            error /= 2
            # if the total error is below the tollerence, then go to the next data
            # else, update the weights and threshold for the output layer first then for the hidden layer
            if error < self.errorTolerance:
                index += 1
            else:
                for node in OutputLayerNode:
                    node.getUpdate()

                for i in range(self.nHiddenNode):
                    HiddenLayerNode[i].getUpdate(sum([node.getErrorWeight(i) for node in OutputLayerNode]))
                index = 0

    def test(self, testData):
        predictedClass = []
        testData = np.array(testData)
        for data in testData:
            predictedClass.append(self.predict(data))
        return predictedClass

    def predict(self, data):
        classLabel = ''
        for node in HiddenLayerNode:
            node.sendData(data)
        for i in range(self.nOutputNode):
            OutputLayerNode[i].sendData([n.output for n in HiddenLayerNode])
            classLabel += str(round(OutputLayerNode[i].output))

        return int(classLabel[::-1], 2)


class HiddenNode:

    def __init__(self, nFeature, eta=0.2, k=0.2):
        self.eta = eta
        self.k = k
        self.threshold = round(random.random(), 2)
        self.nFeature = nFeature
        self.weights = np.array([round(random.random(), 2) for _ in range(nFeature)])

    def sendData(self, data):
        self.data = data
        active = sum(data * self.weights) + self.threshold
        self.output = 1 / (1 + math.exp(-self.k * active))

    def getUpdate(self, summation):
        value = self.eta * self.k * self.output * (1 - self.output) * summation
        self.threshold += value
        for i in range(self.nFeature):
            self.weights[i] += value * self.data[i]


class OutputNode:

    def __init__(self, nHiddenNode, eta=0.8, k=0.8):
        self.eta = eta
        self.k = k
        self.threshold = round(random.random(), 2)
        self.nHiddenNode = nHiddenNode
        self.weights = np.array([round(random.random(), 2) for _ in range(nHiddenNode)])

    def getUpdate(self):
        value = self.eta * self.k * self.error * self.output * (1 - self.output)
        self.threshold += value
        for i in range(self.nHiddenNode):
            self.weights[i] += value * HiddenLayerNode[i].output

    def getErrorWeight(self, index):
        return self.error * self.weights[index]

    def sendData(self, data):
        active = sum(data * self.weights) + self.threshold
        self.output = 1 / (1 + math.exp(-self.k * active))

    def getError(self, actualClass):
        self.error = actualClass - self.output
        return self.error

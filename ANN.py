from numpy import *
from numpy.random import *
from matplotlib import pyplot as pp

class ANN:
    '''A simple NN with 1 hidden layer
        Written By Alexander Saites'''
    def __init__(self,inNeurons,hiddenNeurons,outNeurons,alpha=.01):
        self.inN = inNeurons
        self.hN = hiddenNeurons
        self.oN = outNeurons
        self.alpha = alpha

        self.weight1 = (rand(self.hN,self.inN)-.5)*.1
        self.weight2 = (rand(self.oN,self.hN)- .5)*.1

        self.output1 = None
        self.output2 = None
        self.output3 = None
        self.input2 = None
        self.delta1 = None
        self.delta2 = None

    def FeedForward(self,inVect):
        self.output1 = inVect.reshape(self.inN,1)
        self.input2 = dot(self.weight1, self.output1)
        self.output2 = 1.7159*tanh((2.0/3.0)*self.input2)
        self.output3 = dot(self.weight2, self.output2)

    def UpdateWeights(self,target):
        self.delta2 = self.output3 - target.reshape(self.oN,1)
        self.update2 = self.alpha * dot(self.delta2, self.output2.T)
        self.delta1 = 0.38852302970258*(2.94431281-self.output2*self.output2)\
                        * dot(self.weight2.T, self.delta2)
        self.update1 = self.alpha * dot(self.delta1, self.output1.T)
        self.weight1 -= self.update1
        self.weight2 -= self.update2

    def Train(self, data, labels, numTimes=1):
        targets = identity(self.oN) * 2 - 1
        for i in xrange(numTimes):
            for idx in permutation(data.shape[0]):
                self.FeedForward(data[idx])
                self.UpdateWeights(targets[labels[idx]])

    def Confuse(self, data, labels):
        conf = zeros((self.oN, self.oN))
        miss = 0
        for idx, d in enumerate(data):
            self.FeedForward(d)
            winner = argmax(self.output3)
            conf[labels[idx], winner] += 1
            if labels[idx] != winner:
                miss += 1
        return (conf, miss)

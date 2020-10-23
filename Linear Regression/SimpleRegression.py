import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model

class SimpleRegression:
    def __init__(self, data, head):
        self.data = pd.read_csv("FuelConsumption.csv")
        self.cdf = self.data[head]
        self.msk = None
        self.theta = self.theta0 = None
        self.trainData = self.trainX = self.trainY = None
        self.dataIsTrained = False

    def train(self, dataX, dataY, trainProcentage):
        self.msk = self.maskData(trainProcentage)

        self.trainData = self.cdf[self.msk]

        linReg = linear_model.LinearRegression()
        self.trainX = np.asanyarray(self.trainData[[dataX]])
        self.trainY = np.asanyarray(self.trainData[[dataY]])

        linReg.fit(self.trainX, self.trainY)

        self.theta = linReg.coef_
        self.theta0 = linReg.intercept_

        self.dataIsTrained = True

    def maskData(self, trainProcentage):
        return np.random.rand(len(self.data)) < trainProcentage

    def plotTrainedData(self, dataX, dataY, graphColor = 'blue'):
        if(self.dataIsTrained is True):
            plt.scatter(getattr(self.trainData, dataX), getattr(self.trainData,dataY), color =graphColor)
            plt.plot(self.trainX, self.theta[0]*self.trainX + self.theta0[0], '-r')
            plt.xlabel(dataX)
            plt.ylabel(dataY)
            plt.show()
        else:
            print("Data is not trained yet!")



data = pd.read_csv("FuelConsumption.csv")
head = ['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']

test = SimpleRegression(data, head)
test.train('ENGINESIZE', 'CO2EMISSIONS', 0.8)
test.plotTrainedData('ENGINESIZE', 'CO2EMISSIONS')

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score



class SimpleRegression:
    def __init__(self, data, head):
        self.data = pd.read_csv("FuelConsumption.csv")
        self.cdf = self.data[head]
        self.linReg = None
        self.theta = self.theta0 = None
        self.trainData = self.trainX = self.trainY = None
        self.testData = None
        self.dataIsTrained = False

    def train(self, dataX, dataY, trainProcentage):
        msk = self.maskData(trainProcentage)

        self.trainData = self.cdf[msk]
        self.testData = self.cdf[~msk]

        self.linReg = linear_model.LinearRegression()
        self.trainX = np.asanyarray(self.trainData[[dataX]])
        self.trainY = np.asanyarray(self.trainData[[dataY]])

        self.linReg.fit(self.trainX, self.trainY)

        self.theta = self.linReg.coef_
        self.theta0 = self.linReg.intercept_

        self.dataIsTrained = True

    def maskData(self, trainProcentage):
        return np.random.rand(len(self.data)) < trainProcentage

    def printError(self, dataX, dataY):
        test_x = np.asanyarray(self.testData[[dataX]])
        test_y = np.asanyarray(self.testData[[dataY]])
        test_y_ = self.linReg.predict(test_x)

        print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
        print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
        print("R2-score: %.2f" % r2_score(test_y , test_y_) )

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
test.printError('ENGINESIZE', 'CO2EMISSIONS')
test.plotTrainedData('ENGINESIZE', 'CO2EMISSIONS')

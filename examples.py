from simplesom import *
from utils import *

colors = array([[0,0,0],
                [1,0,0],
                [0,1,0],
                [0,0,1],
                [1,0,1],
                [1,1,0],
                [0,1,1],
                [1,1,1]])

print('Create a 2D SOM and train it using colors')
s = SOM(numInputs=3, shape=(100,100), alphaInit=.5)
runData(s, colors, numTimes=125)
simpleView(s, (1,1), color=True)

print('Create a 2D SOM and train it using a function for data')
s = SOM(3,(100,100))
runData(s, lambda i : rand(3,1), numTimes=1000)
simpleView(s, (1,1), color=True)

print('Create a SOM and animate its training using colors')
s = SOM(3,(100,100))
animateTraining(s, colors, imshape=(1,1), color=True, numTimes=125)

print('Create a SOM and animate its training using a function for data')
s = SOM(3,(100,100))
animateTraining(s, lambda i : rand(3,1), imshape=(1,1), \
    color=True, numTimes=1000)

print('Create a 3D SOM and train it using a function for data')
s = SOM(3,(100,100,100))
runData(s, lambda i : rand(3,1), numTimes = 1000)
simpleView(s, (1,1), color=True) #exactly as you would in the 2D case


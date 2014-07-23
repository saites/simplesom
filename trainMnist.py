from numpy import *
from simplesom import *
from utils import *
from ANN import *
import random

PATH = 'MNISTData'

# get images
data = load(PATH+'/train.npy')
labels = data[:,0]
train_images = data[:,1:]

# normalize 
train_images = train_images / 255.0 

smallSet = train_images[:10000,:].reshape(10000,28,28)
l = labels[:10000]

# build and train soms 
som = SOM(28*28, shape=(10,10))
animateTraining(som, smallSet, imshape=(28,28))

bmus = getBMUs(som, smallSet)
net = ANN(2,20,10)
net.Train(bmus, l, numTimes=5)
conf, miss = net.Confuse(bmus, l)
print conf
print float(miss)/float(bmus.shape[0])

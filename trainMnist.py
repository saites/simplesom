from numpy import *
from simplesom import *
from utils import *
from ANN import *
import random
import cPickle as pk

set_printoptions(suppress=True)
PATH = 'MNISTData'

# get images
data = load(PATH+'/train.npy')
labels = data[:,0]
train_images = data[:,1:]

# normalize 
train_images = train_images / 255.0 

nIms = 40000
smallSet = train_images[:nIms,:].reshape(nIms,28,28)
l = labels[:nIms]

# build and train soms 
som = SOM(28*28, shape=(14, 14, 14)) 
#animateTraining(som, smallSet, imshape=(28,28))
print "running data"
runData(som, smallSet, numTimes=4)
#simpleView(som, imshape=(28,28))
pk.dump(som, open('3dsom.p', 'wb'))

print "training network"
bmus = getBMUs(som, smallSet)
net = ANN(len(som.som.shape)-1,20,10)
net.Train(bmus, l, numTimes=7)
conf, miss = net.Confuse(bmus, l)
print conf
print float(miss)/float(bmus.shape[0])
pk.dump(net, open('net.p', 'wb'))

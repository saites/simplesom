A Self Organizing Map Library for Topological Analysis
Alexander Saites
July 2014

This library is designed to simplify the process of creating and analyzing self
organizing maps in python. It includes several utility functions for creating
and training SOMs. It is free to use with citation.

--------------
Dependencies
--------------
SimpleSOM makes heavy use of numpy and matplotlib. You should install scipy
before using SimpleSOM.


--------------
Quick Guide
--------------
Documentation is included with the source, but this will give you a quick 
overview.

####Creating a SOM
`s = som(numInputs=3, shape=(10,10,10), alphaInit=.6)`

Training a SOM manually
    s.setLearnParams(numIterations=1000)
    for d in data:
        s.findBMU(d)
        s.updateSOM()
        s.updateLearnParams()

Training a SOM automatically, using utils
    from utils import runData
    runData(s, data, 1000)

Viewing the training of a 2D SOM
    animateTraining(s, data, imshape=(28,28))

Viewing the activation vectors of a 2D or 3D SOM
    simpleView(s, imshape=(28,28))

Viewing the strength of activations for a 2D SOM
    fullView(s, data, data)

Viewing the activations of a 2D SOM for which the input is not an image
    fullView(s, data, imageSet)

Viewing the activations of a 2D SOM for which there is only one image per class
    fullView(s, data, images, labels)

Training a simple NN using ANN and showing a confusion matrix
    from ANN import *
    from utils import *

    runData(s, data, 1000)
    bmus = getBMUs(s, data)
    net = ANN(len(s.som.shape)-1, 100, numClasses)
    net.Train(bmus, labels, numTimes=7)
    confusionMatrix, misses = net.Confuse(bmus, labels)
    print confusionMatrix
    print float(misses)/float(bmus.shape[0])


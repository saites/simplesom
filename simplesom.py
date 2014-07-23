from numpy import *
from numpy.random import rand,permutation,random_sample

class SOM:
    '''
    Simple multidimensional SOM, written by Alexander Saites
    '''
    def __init__(self, numInputs, shape, alphaInit=.5):
        '''
        Initize the som with a given shape
            numInputs - int
                The dimensionality of the input space.
            shape - a tuple of ints
                The shape of the network.
            alphaInit - float
                The initial alpha value for training.
                Alpha controls the height of the guassian used to update the 
                neurons about the best matching unit. The value should be in 
                [0,1] The default value is .5, and it slowly decays over time.
        '''
        if any(shape < 0):
            raise ValueError('shape values must be greater than 0')

        self.dim = len(shape)
        self.som = random_sample(shape + (numInputs,))
        self.nIn = numInputs
        self.alphaInit = alphaInit
        self.setLearnParams(1)

        # These grids simplify calculating distances later on.
        # Each element of the list is a range reshaped to one of the
        # dimensions of our shape. We'll subtract the location of our
        # winning neuron, square the result, and add them all together
        # to get the euclidean distance of each neuron from the winner
        self.grids = [arange(shape[i]).reshape(\
            tuple([1]*i + [shape[i]] + [1]*(self.dim-i-1)))\
            for i in xrange(self.dim)]

    def setLearnParams(self, numIterations):
        '''
        Set the initial learning parameters based on the number of 
        iterations we intend to perform.
            numIterations - int
                The number of iterations that will be performed. That is, if
                you call setLearnParams(100), after 100 calls to
                updateLearnParams, the learning parameters will have decayed to
                their lower value.
        '''
        self.r = float(max(self.som.shape[:-1]))
        self.delta = float(numIterations) / float(max(log(self.r),1))
        self.alpha = self.alphaInit
        self.curIter = 0
        self.omega2 = self.r * self.r

    def findBMU(self, inVect):
        '''
        Find the best matching unit for this input.
            inVect - numpy ndarry
                An input vector.
        '''
        self.diff = inVect.flat - self.som
        self.sqdiff = self.diff*self.diff
        self.sumdiff = sum(self.sqdiff, axis=self.dim)
        self.BMU = unravel_index(argmin(self.sumdiff), self.sumdiff.shape)
        return self.BMU

    def updateSOM(self):
        '''
        Update a portion of the map about the winner.
        '''
        # get eucl distance to winner
        sqdist = sum([(g - self.BMU[i]) * (g - self.BMU[i])\
            for i,g in enumerate(self.grids)])
        mask = sqdist <= self.omega2

        # update guassian about BMU
        update = exp(-sqdist/(2*self.omega2))
        self.som[mask] += self.alpha * update[...,None][mask]*self.diff[mask]
        self.som[self.som > 1] = 1
        self.som[self.som < 0] = 0

    def updateLearnParams(self):
        '''
        Update the learning parameters for the next iteration.
        '''
        self.curIter += 1
        self.alpha = self.alphaInit * exp(-self.curIter/self.delta)
        omega = self.r * exp(-self.curIter/self.delta)
        self.omega2 = omega * omega

if __name__ == '__main__':
    from matplotlib import pyplot as pp
    from matplotlib import animation as ani
    from utils import runData, simpleView

    colors = array([[0,0,0],
                    [1,0,0],
                    [0,1,0],
                    [0,0,1],
                    [1,0,1],
                    [1,1,0],
                    [0,1,1],
                    [1,1,1]])

    som = SOM(3,shape=(20,20,20))
    runData(som, colors, 1000)
    simpleView(som, shape=(1,1), color=True)

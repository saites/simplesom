from numpy import *
from matplotlib import pyplot as pp
from matplotlib.animation import FuncAnimation
from numpy.random import permutation

def imreshape(image,newshape):
    '''
    Reshapes flattened activation vectors to their proper 2D counterpart by 
    arranging image squares of the sizes of the images.
        image - numpy ndarry
            The flattened activation vector.
        newshape - tuple of ints
            A tuple of the form (H,W,Ih,Iw,[3]), where H,W are the height and
            width of the original SOM; Ih, Iw are the height and width of the
            images, and 3 is an optional parameter that should be specified if
            the images are meant to be in color. If the original images were
            simply intensity values, then submit only (H,W,Ih,Iw).
    '''
    return concatenate(concatenate(image.reshape(newshape),axis=2),axis=0)

def animateTraining(som, data, imshape, color=False, numTimes=1, fig=None):
    '''
    Show the SOM at each step of training
        som - a SOM object
        data - a numpy object
            the dataset to use for training
        imshape - a 2-tuple of ints
            the shape of the images used for training
        color - boolean, default: False
            if True: images are assumed to have three color channels
            if False: images are assumed to be intensity only
        numTimes - integer, default: 1
            number of times to train over the dataset
        fig - a matplotlib.pyplot figure object, default: None
            the figure that will be used for display
            if None, a new figure will be created
    '''
    # check inputs
    if len(som.som.shape) != 3:
        raise ValueError('animation training only works for 2D soms')
    if not isinstance(data, ndarray) and not callable(data):
        raise TypeError('data should be an array or callable')
    if len(imshape) != 2:
        raise ValueError('imshape should be a two-tuple')

    # set values based on parameters
    H,W,D = som.som.shape
    fig = pp.figure() if not fig else fig
    dataSize = 1 if callable(data) else data.shape[0]
    newshape = (H,W)+imshape+(3,) if color else (H,W)+imshape
    im = pp.imshow(imreshape(som.som, newshape))
    som.setLearnParams(dataSize * numTimes)

    # define functions for animation
    def animate(i):
        global __som_d_order #need this to be static
        if callable(data):
            som.findBMU(data(i))
        else:
            if i % dataSize == 0:
                __som_d_order = permutation(range(dataSize))
            som.findBMU(data[__som_d_order[i % dataSize]])
        som.updateSOM()
        som.updateLearnParams()
        im.set_data(imreshape(som.som,newshape))
        return [im]

    anim = FuncAnimation(fig, animate, frames = dataSize*numTimes, 
        interval = 20, blit=True, repeat=False)
    pp.show()

def runData(som, data, numTimes=1):
    '''
    Trains a SOM using a particular dataset.
        som - a SOM object 
        data - a numpy array 
            the data to train on
        numTimes - an integer
            the number of times to train over the data
    '''
    if not isinstance(data, ndarray) and not callable(data):
        raise TypeError('data should be an array or callable')
        
    if callable(data):
        dataSize = 1
    else:
        dataSize = data.shape[0]
    som.setLearnParams(dataSize * numTimes)
    for i in xrange(numTimes):
        for j in permutation(range(dataSize)):
            if callable(data):
                som.findBMU(data(i*dataSize+j))
            else:
                som.findBMU(data[j])
            som.updateSOM()
            som.updateLearnParams()

def simpleView(som, imshape, color=False):
    '''
    Uses imshow to show the activation vectors of a 2D SOM, or to animate 
    through the 2D slices of the activation vectors of a 3D SOM.
        som - a SOM object
        imshape - 2-tuple of ints
            the shape of each image
        color - boolean, default: False
            if true, expects images to be in color
            if false, expects images to be intensity only
    '''
    dim = len(som.som.shape)-1
    if dim not in [2,3]: 
        raise ValueError('simpleview only works for 2D and 3D soms')

    if dim == 2:
        H,W,_ = som.som.shape
    else:
        D,H,W,_ = som.som.shape
    newshape = (H,W)+imshape+(3,) if color else (H,W)+imshape

    fig = pp.figure()

    if dim == 2:
        im = pp.imshow(imreshape(som.som, newshape))
    else:
        im = pp.imshow(imreshape(som.som[0,...], newshape))
        def animate(i=0):
            im.set_data(imreshape(som.som[i,...],newshape))
            return [im]
        anim = FuncAnimation(fig,animate,frames=D,interval=200,blit=True)
    pp.show()

def fullView(som, data, images, labels=None):
    '''
    Maps images to the BMUs of a SOM based on their associated data vectors.
    If labels is None, then data and images should have the same length, as it
    is assumed that data[i] maps to images[i]. If labels is used, then it 
    should have the same length as data, and it is assumed that data[i] maps to
    images[labels[i]].
        som - a SOM object
        data - a numpy ndarray
            The data set used to train the map.
        images - a numpy ndarray
            The images to map to the BMUs.
        labels - a numpy ndarray, default: None
            An array of labels. If used, BMU[data[i]] = images[labels[i]].
    '''
    if labels is not None and data.shape[0] != labels.shape[0]:
        raise ValueError('data and labels should have the same length')
    elif labels is None and data.shape[0] != images.shape[0]:
        raise ValueError('data and image should have the same length')

    dim = len(som.som.shape)-1
    if dim not in [2]:
        raise ValueError('fullView only works for 2D soms')

    H,W,_ = som.som.shape
    color = len(images[0].shape) == 3
    imH, imW = images[0].shape[:2]
    imshape = (H*imH,W*imW) + images[0].shape[2:]
    view = zeros(imshape)

    for i,d in enumerate(data):
        (y,x) = som.findBMU(d)
        view[(x * imW):((x+1) * imW), (y * imH):((y+1) * imH), ...] += \
            images[labels[i]] if labels is not None else images[i]

    view /= data.shape[0]
    pp.imshow(view)
    pp.show()

def getBMUs(som, data):
    '''
    Returns the best matching unit for each item in the dataset.
        som - a SOM object
        data - a numpy ndarray
    '''
    return array([som.findBMU(d) for d in data])

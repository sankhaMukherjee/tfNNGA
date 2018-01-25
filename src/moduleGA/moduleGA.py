from logs import logDecorator as lD 
from lib.libGA import GA
import json

import numpy      as np
import tensorflow as tf

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.moduleGA'


@lD.log(logBase + '.testGA')
def testGA(logger):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {[type]}
        [description]
    '''

    try:
        GAconfig = json.load(open('../config/GA.json'))
    except Exception as e:
        logger.error('Unable to obtain the configuration for GA')
        return

    # Let us start with a simple linear regression problem. 
    # Equation:
    #     y = WX 
    # The shapes that we have planned ...
    #     (1, None)  = (1, 3) x (3, None)
    # In this specific case, W = [0.5, 0.2, 1]

    Warr = np.array([[0.1, 0, 0]])
    Xarr = np.random.random((3, 1000))
    yarr = np.dot(Warr, Xarr)

    # W  = tf.convert_to_tensor(np.random.rand(1, 3), dtype=tf.float32)
    W  = tf.Variable(tf.convert_to_tensor(Warr, dtype=tf.float32))
    X  = tf.placeholder(dtype=tf.float32, shape=(3, None))
    y  = tf.placeholder(dtype=tf.float32, shape=(None) )

    yHat  = tf.matmul(W, X)
    error = tf.reduce_mean( tf.sqrt((y - yHat)*(y - yHat)) )

    variables    = [W]
    costFunction = error

    simpleGA = GA.GA(variables, costFunction, GAconfig, X, y,'initType')
    simpleGA.findPopulationCosts(Xarr, yarr)
    print(simpleGA)

    if True:
        print('\nThis is for testing the crossover ...')
        simpleGA.printCurrErr()
        for i in range(30):
            simpleGA.crossover(Xarr, yarr)
            # print('Crossover:') 
            # simpleGA.printCurrErr()

            simpleGA.mutation(Xarr, yarr)
            # print('Mutation:') 
            # simpleGA.printCurrErr()

            print('Error: {}, Best Gene:{}'.format(simpleGA.printCurrErr(), simpleGA.printBestGene()), flush=True)
            

    if False:
        print('\nThis should give low error ...')
        print(simpleGA.findError(Xarr, yarr))

        print('\nThis should give an error of approximately 1 ...')
        print(simpleGA.findError(Xarr, yarr + 1))
        
        print('\nThis should give an error of approximately 2 ...')
        print(simpleGA.findError(Xarr, yarr + 2))

    if False:
        print('\n This is using the fit function')
        print(simpleGA.fit(Xarr, yarr))

    return

@lD.log(logBase + '.initGA')
def initGA(logger):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {[type]}
        [description]
    '''

    try:
        GAconfig = json.load(open('../config/GA.json'))
    except Exception as e:
        logger.error('Unable to obtain the configuration for GA')
        return

    print('We are in the GA module')
    return

@lD.log(logBase + '.main')
def main(logger):
    '''main function for module1
    
    This function finishes all the tasks for the
    main function. This is a way in which a 
    particular module is going to be executed. 
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger function
    '''

    testGA()
    initGA()

    return


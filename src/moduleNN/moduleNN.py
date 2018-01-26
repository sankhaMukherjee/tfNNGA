from logs import logDecorator as lD 
import json, os

import tensorflow        as tf
import numpy             as np
import matplotlib.pyplot as plt

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.moduleNN'


@lD.log(logBase + '.generateNN')
def generateNN(logger):
    '''Optimize a neural network
    
    This generates a very simple neural network structure and
    optimize it. 

    Parameters
    ----------
    logger : {[type]}
        [description]
    '''

    # We shall generate a fairly simple function and attempt to
    # see whether the neural network generates a model that can
    # can be easily learned with normal gradient decent. The
    # problem we are modeling is the following:
    # Input: [x1, x2]
    # Output: y = 2*sin(x1) + 3*cos(x2)
    # we shall generate i/o values for:
    #    -np.pi/2 <= x1 <= np.pi
    #    -np.pi/2 <= x2 <= np.pi

    Xarr = np.random.rand(2, 10000) * np.pi - np.pi/2
    # yarr = 2*np.sin(Xarr[0,:]) + 3*np.cos(Xarr[1,:])
    yarr = 2*Xarr[0,:] + 3*Xarr[1,:]

    # This is for saving the parameter space. 
    if False:
        plt.scatter( Xarr[0, :], Xarr[1, :], c=yarr, alpha=0.1 )
        plt.savefig(os.path.join(config['results']['resultsImgFolder'], 'surf.png'))


    # Model this with a 3 layer network. 
    # [2, 1000] -> [10, 1000] -> [5, 1000] -> [1, 1000]
    decimator = 0.1

    inp = tf.placeholder(dtype=tf.float32, shape=(2, None))
    out = tf.placeholder(dtype=tf.float32, shape=(1, None))
    W1  = tf.Variable(tf.convert_to_tensor( np.random.rand( 10, 2 ) * decimator, dtype=tf.float32 ))
    W2  = tf.Variable(tf.convert_to_tensor( np.random.rand( 5, 10 ) * decimator, dtype=tf.float32 ))
    W3  = tf.Variable(tf.convert_to_tensor( np.random.rand( 1, 5 )  * decimator, dtype=tf.float32 ))


    v1 = tf.keras.activations.tanh( tf.matmul( W1, inp) )
    v1 = tf.keras.activations.tanh( tf.matmul( W2, v1) )
    v1 = tf.matmul( W3, v1) # This is a regression problem

    error  = tf.reduce_mean(  (v1 - out)**2  )
    sqrErr = tf.sqrt( error )

    opt = tf.train.AdamOptimizer(learning_rate=0.1).minimize( sqrErr )

    results = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(10000):
            _, result = sess.run( [opt, sqrErr], feed_dict = {
                inp: Xarr, out: yarr.reshape(1, -1)
                })

            results.append(result)
            if i%500 == 0:
                print(result, flush=True)

        yHat = sess.run(v1, feed_dict = {
                inp: Xarr, out: yarr.reshape(1, -1)
                })
    

    plt.figure()
    plt.plot(results)
    plt.yscale('log')
    plt.savefig(os.path.join(config['results']['resultsImgFolder'], 'error.png'))

    plt.figure()
    plt.plot(yarr, yHat[0, :], 's', alpha=0.1 )
    plt.savefig(os.path.join(config['results']['resultsImgFolder'], 'values.png'))

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

    generateNN()

    return


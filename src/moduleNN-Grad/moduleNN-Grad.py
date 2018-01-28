from logs import logDecorator as lD 
from tqdm import tqdm
import json, os

import tensorflow        as tf
import numpy             as np
import matplotlib.pyplot as plt

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.moduleNN-Grad'


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

    print('#'*50)
    print('We are in NN-Grad ...')

    # This is for saving the parameter space. 
    if False:
        plt.scatter( Xarr[0, :], Xarr[1, :], c=yarr, alpha=0.1 )
        plt.savefig(os.path.join(config['results']['resultsImgFolder'], 'surf.png'))


    # Model this with a 3 layer network. 
    # [2, 1000] -> [10, 1000] -> [5, 1000] -> [1, 1000]
    decimator      = 0.1
    regularization = 1e-4
    iterations     = 1000

    inp = tf.placeholder(dtype=tf.float32, shape=(2, None))
    out = tf.placeholder(dtype=tf.float32, shape=(1, None))
    W1  = tf.Variable( tf.convert_to_tensor( (np.random.rand( 10, 2 )  - 0.5) * decimator , dtype=tf.float32 ))
    W2  = tf.Variable( tf.convert_to_tensor( (np.random.rand( 5, 10 )  - 0.5) * decimator , dtype=tf.float32 ))
    W3  = tf.Variable( tf.convert_to_tensor( (np.random.rand( 1, 5 )  - 0.5)  * decimator , dtype=tf.float32 ))

    Ws = [W1, W2, W3]

    v1 = tf.keras.activations.tanh( tf.matmul( W1, inp) )
    v1 = tf.keras.activations.tanh( tf.matmul( W2, v1) )
    v1 = tf.matmul( W3, v1) # This is a regression problem

    # reg = np.mean([tf.reduce_mean(W**2) for W in [W1, W2, W3]]) # L2 regularization
    regL2 = tf.reduce_mean(tf.convert_to_tensor([tf.reduce_mean(W**2) for W in Ws ]))

    error  = tf.reduce_mean(  (v1 - out)**2  )
    sqrErr = tf.sqrt( error + regularization*tf.convert_to_tensor(regL2) )

    opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize( sqrErr )

    results = []
    deltas  = []
    stds    = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        oldVals = sess.run(Ws)

        for i in tqdm(range(iterations)):
            _, result = sess.run( [opt, sqrErr], feed_dict = {
                inp: Xarr, out: yarr.reshape(1, -1)
                })

            results.append(result)
            results.append(result)
            newVals = sess.run(Ws)
            deltas.append( [(o-n).mean() for o, n in zip(oldVals, newVals)] )
            stds.append( [(o-n).std() for o, n in zip(oldVals, newVals)] )
            oldVals = newVals.copy()

            if i%500 == 0:
                tqdm.write(str(result))

        yHat = sess.run(v1, feed_dict = {
                inp: Xarr, out: yarr.reshape(1, -1)
                })

        print(sess.run([W1, W2, W3]))

    # print(deltas)
    

    plt.figure()
    plt.plot(results)
    plt.yscale('log')
    plt.savefig(os.path.join(config['results']['resultsImgFolder'], 'error.png'))

    plt.figure()
    plt.plot(yarr, yHat[0, :], 's', alpha=0.1 )
    plt.savefig(os.path.join(config['results']['resultsImgFolder'], 'values.png'))

    plt.figure()
    for i, (d, s) in enumerate(zip(np.array(deltas).T, np.array(stds).T)):
        plt.plot(d, label='{}'.format(i) )
        plt.fill_between(d-s, d+s, alpha=0.1, label='{}'.format(i) )
    plt.legend()
    plt.savefig(os.path.join(config['results']['resultsImgFolder'], 'deltas.png'))

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


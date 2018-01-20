from tqdm import tqdm
from logs import logDecorator as lD
import json

import numpy      as np
import tensorflow as tf

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.libGA.GA'

class GA():

    @lD.log(logBase + '.__init__')
    def __init__(logger, self, variables, costFunction, GAconfig, X, y, name='testGA'):
        '''initializer for the GA class
        
        Initialize the genes that are going to be doing the optimization
        and also set the initial parameter for the hyperparameters. 
        
        Decorators:
            lD.log
        
        Arguments:
            logger {logging.Logger} -- logging object. This should not
                be passed to this iniitalizer. This will be inserted
                into the function directly form the decorator. 
            self {instance} -- variable for the instance of the GA class
            variables {list of tensors} -- This is a list of tensors that we
                want to train for the optimization. 
            costFunction {tensor} -- A 0-dimensional tensor that needs to be 
                optimized. At this moment, we will assume that this is to be
                minimized.
            GAconfig {dict} -- configuration dict that will contain details
                for all the hyperparameters for the genetic algorithm. 
            X {tf.placeholder} -- The placeholder for the input data
            y {tf.placeholder} -- The placeholder for the output data
        
        Keyword Arguments:
            name {str} -- this is the name of the instance. Use the name to print
                information about the GA instance that you are using (default: 
                {'testGA'})
        '''

        self.properConfig = False
        self.currentErr   = np.nan

        try:
            self.name         = name
            self.GAconfig     = GAconfig
            self.variables    = variables
            self.costFunction = costFunction
            self.X            = X
            self.y            = y

            # Now we shall generate the population
            # Note that how this is going to be generates will 
            # be done later ... 
            # ----------------------------------------------------
            print('Generating initial population ...')
            self.population   = []
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                tempVars = []
                for i in tqdm(range(self.GAconfig['numChildren'])):
                    temp = [(v + (np.random.random( v.shape ) - 0.5) * 2) for v in self.variables]
                    sess.run(temp)
                    tempVars.append( temp )

                self.population.append( tempVars )

            assert type(self.name) == str

        except Exception as e:
            logger.error('Unable to generate the proper struclture of the GA: {}'.format(
                str(e)))
            return

        # Here we say that the GA was properly configured
        self.properConfig = True

        return

    @lD.log(logBase + '.__repr__')
    def __repr__(logger, self):
        '''allows you to print this instance
        
        This is a very simple configutation. This is going to show what 
        the instance looks like at a particular point. 
        
        Decorators:
            lD.log
        
        Arguments:
            logger {logging.Logger} -- logging object. This should not
                be passed to this iniitalizer. This will be inserted
                into the function directly form the decorator. 
            self {instance} -- variable for the instance of the GA class
        '''

        if not self.properConfig:
            logger.error('Unable to print a badly formed class')
            return


        result = '------------[ {:20s} ]-----------------\n'.format(self.name)

        for k in self.GAconfig:
            result += '{:30s}: {}\n'.format( k, self.GAconfig[k] )
        result += '.'*30 + '\n'
        result += 'Number of variables: {}\n'.format(len(self.variables))
        for i, v in enumerate(self.variables):
            result += '\tShape of variable {:5d} = {}\n'.format(i, v.shape)
        result += '.'*30 + '\n'
        result += ' Characteristics of the population ...\n'
        result += 'population variable value\n'

        for i, p in enumerate(self.population):
            for j, variables in enumerate(p):
                for k, v in enumerate(variables):
                    result +=  '{:10} {:8} {:5} --> {}\n'.format(i, j, k, str(v.shape))
        result += '.'*30 + '\n'


        return result

    @lD.log(logBase + '.findError')
    def findError(logger, self, X, y):
        '''returns the error function for the current condition
        
        This is the condition wherein we can check of the supplied
        weights are able to generate an exepcted error. This funciton
        is a very quick and easy way of checking whether the supplied
        costFunction is able to be computed from the provided variables.

        This is always going to assume that the variables 
        
        Decorators:
            lD.log
        
        Arguments:
            logger {logging.Logger} -- logging object. This should not
                be passed to this iniitalizer. This will be inserted
                into the function directly form the decorator. 
            self {instance} -- variable for the instance of the GA class
            X {numpy.array} -- The set of input values to be tested
            y {numpy.array} -- The expected results
        '''

        result = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run(self.costFunction, feed_dict={
                self.X : X, self.y : y
                })

        return result

    @lD.log(logBase + '.fit')
    def fit(logger, self, X, y):
        '''[summary]
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger {[type]} -- [description]
            self {[type]} -- [description]
            X {[type]} -- [description]
            y {[type]} -- [description]
        '''

        print('======[ Session 1   ]===================')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            print('--------- This is before the addition ----------------')
            print(sess.run(self.variables))

            print('--------- This is during the addition ----------------')
            vNew = []
            for i, v in enumerate(self.variables):
                v = v + 1
                print(sess.run( v ))
                vNew.append(v)
            self.variables = vNew

            print('--------- This is after the addition ----------------')
            print(sess.run(self.variables))

        print('======[ Session 2   ]===================')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('This is to make sure that the variables span across sessions ...')
            print(sess.run(self.variables))
            print('This is to make sure that these are independent variables ...')
            print(sess.run(self.population[0][0] ))



        return 





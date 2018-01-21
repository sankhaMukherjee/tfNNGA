from tqdm import tqdm
from logs import logDecorator as lD
import json, os

import numpy             as np
import tensorflow        as tf
import matplotlib.pyplot as plt

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
        self.currentErr   = None

        try:
            self.name         = name
            self.GAconfig     = GAconfig
            self.variables    = variables
            self.costFunction = costFunction
            self.X            = X
            self.y            = y

            # Now we shall generate the population
            print('Generating initial population ...')
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                locVars = sess.run(self.variables)
            
            # Now we generate new variables directly from the old ones
            # Note that how this is going to be generates will 
            # be done later ... 
            # ----------------------------------------------------
            self.population   = []
            for i in tqdm(range(self.GAconfig['numChildren'])):
                temp = []
                for j, v in enumerate(locVars):
                    v = (v + (np.random.random( v.shape ) - 0.5) * 2)
                    temp.append(v)
                self.population.append( temp )

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
        result += '.'*30 + '\n'
        result += ' Characteristics of the error ...\n'
        result += 'quan:value\n'
        if self.currentErr is not None:
            result += '   0:{:.4}\n  10:{:.4}\n  25:{:.4}\n  50:{:.4}\n  75:{:.4}\n  90:{:.4}\n 100:{:.4}\n'.format(*np.percentile(self.currentErr, [0, 10, 25, 50, 75, 90, 100]))

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

        if not self.properConfig:
            logger.error('Unable to generate the current errors due to improper configuration')
            return

        result = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run(self.costFunction, feed_dict={
                self.X : X, self.y : y
                })

        return result

    @lD.log(logBase + '.testVariableChange')
    def testVariableChange(logger, self, X, y):
        '''[summary]
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger {logging.Logger} -- logging object. This should not
                be passed to this iniitalizer. This will be inserted
                into the function directly form the decorator. 
            self {instance} -- variable for the instance of the GA class
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

    @lD.log(logBase + '.findCosts')
    def findPopulationCosts(logger, self, X, y):
        '''finds the cost of the entire population. 
        
        This is similar to the function findError except that this
        function finds the cost to the entire population rather than
        the cost to the entire population that we are trying to
        generate rather than only the cost of the single variable. This
        function is also going to update the self.currentError variable.
        
        Decorators:
            lD.log
        
        Arguments:
            logger {logging.Logger} -- logging object. This should not
                be passed to this iniitalizer. This will be inserted
                into the function directly form the decorator. 
            self {instance} -- variable for the instance of the GA class
            X {numpy.array} -- The set of input values to be tested
            y {numpy.array} -- The expected results
        
        Returns:
            numpy.array -- the cost calculated for each of the units in the
                current population. If there is an error, this is going to
                return None. 
        '''
        results = None
        self.currentErr = None

        try:
            results = []
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                # print(self.population)
                for ps in self.population:
                    for i, v in enumerate(self.variables): 
                        sess.run(tf.assign( self.variables[i], ps[i] ))

                    result = sess.run(self.costFunction, feed_dict={
                                            self.X : X, self.y : y
                                            })
                    results.append(result)

            self.currentErr = results

        except Exception as e:
            logger.error('Unable to generate the costs of the current population: {}'.format(str(e)))

        return results

    @lD.log(logBase + '.crossover')
    def crossover(logger, self, X, y):
        '''create a crossover for generating a new population
        
        The crossover function is created. This is going to assume that
        the population cost has already been generated. calling this step
        once will result in the generation of a new population. 
        
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

        if not self.properConfig:
            logger.error('The instance is not properly initialized')
            return

        normalize = np.array(self.currentErr).copy()
        normalize = normalize / normalize.max()
        normalize = 1 - normalize
        normalize = normalize / normalize.sum()

        # This stuf is for testing ...
        if False:
            print(list(zip(normalize, self.currentErr)))
            plt.plot(normalize, self.currentErr, 's', mfc='None', mec = 'blue', alpha = 0.5)
            plt.savefig(os.path.join(config['results']['resultsImgFolder'], 'testNormalize.png'))


        choices = np.random.choice( range(len(self.currentErr)), size=(100, 2) , p=normalize )
        print(choices)


        return

    @lD.log(logBase + '.fit')
    def fit(logger, self, X, y):
        '''[summary]
        
        [description]
        
        Decorators:
            lD.log
        
        Arguments:
            logger {logging.Logger} -- logging object. This should not
                be passed to this iniitalizer. This will be inserted
                into the function directly form the decorator. 
            self {instance} -- variable for the instance of the GA class
            X {[type]} -- [description]
            y {[type]} -- [description]
        '''

        if not self.properConfig:
            logger.error('Unable to generate the current errors due to improper configuration')
            return

        results = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # print(self.population)
            for ps in self.population:
                for i, v in enumerate(self.variables): 
                    sess.run(tf.assign( self.variables[i], ps[i] ))

                result = sess.run(self.costFunction, feed_dict={
                                        self.X : X, self.y : y
                                        })
                results.append(result)

        print('A total of {} results obtained ...'.format(len(results)))

        return 






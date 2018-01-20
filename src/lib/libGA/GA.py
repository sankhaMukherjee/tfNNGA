from logs import logDecorator as lD
import json

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.libGA.GA'

class GA():

    @lD.log(logBase + '.__init__')
    def __init__(logger, self, variables, costFunction, GAconfig, name='testGA'):
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
        
        Keyword Arguments:
            name {str} -- this is the name of the instance. Use the name to print
                information about the GA instance that you are using (default: 
                {'testGA'})
        '''

        self.properConfig = False
        try:
            self.name         = name
            self.GAconfig     = GAconfig
            self.variables    = variables
            self.costFunction = costFunction

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

        return result


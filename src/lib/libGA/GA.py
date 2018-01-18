from logs import logDecorator as lD
import json

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.libGA.GA'

class GA():

    def __init__(self):

        return


import pandas as pd
import random

class Network():

    def __init__(self):
        self.layers = []
        self.connections = []
        self.weights = []

        
    def add_layer(self, neurons=5):
        self.layers.append(neurons)

    def initialize_weights(self):
        layer_list_first_excluded = list(self.layers)
        previous_layer = layer_list_first_excluded.pop(0)
        for layer in layer_list_first_excluded:
            self.connections.append(previous_layer * layer)
            previous_layer = layer

        for i in range(sum(self.connections)):
            self.weights.append(random.random())

Network = Network()

Network.add_layer(8)
Network.add_layer(4)
Network.add_layer(2)

Network.initialize_weights()


import random
from math import exp

class Network():

    def __init__(self):
        self.layers = []
        self.connections = []
        self.weights = []
        self.values = []
        self.values_list = []
        self.nodes = 0

    def run(self, dict=[]):
        if len(dict) !=  self.layers[0]:
            raise Exception("Incorrect Shape of Input")
        else:
            self.values[0] = dict

    def activation_func(self, func_name = "", value = 0):
        if (func_name == "Sigmoid"):
           return(1 / 1 - exp(-value))


    def feed_forward(self):
        current_layer = self.values.index([])
        previous_values = self.values[current_layer - 1]
        weights = self.weights[current_layer - 1]
        number_of_neurons_pre_layer = len(previous_values)

        values = []

        for neuron in range(self.layers[current_layer]):
            temp_list = []
            index_position = neuron * number_of_neurons_pre_layer
            temp_weights = weights[index_position: index_position + number_of_neurons_pre_layer]
            for value, weight in zip(previous_values, temp_weights):
                temp_list.append(value * weight)
            values.append(Network.activation_func("Sigmoid", sum(temp_list)))
        self.values[current_layer] = values


    def add_layer(self, neurons=5):
        self.layers.append(neurons)
        self.nodes += 1

    def initialize_weights(self):
        layer_list_first_excluded = list(self.layers)
        previous_layer = layer_list_first_excluded.pop(0)
        for layer in layer_list_first_excluded:
            self.connections.append(previous_layer * layer)
            previous_layer = layer

        for connections in self.connections:
            temp_list = []
            for i in range(connections):
                temp_list.append(random.random())
            self.weights.append(temp_list)


    def train(self, inputs, labels):
        for input in inputs:
            # creates a list in [[],[],[]] format for self.values
            for node in range(self.nodes):
                self.values.append([])

            Network.run(dict=input)
            i = self.nodes - 1
            while i > 0:
                Network.feed_forward()
                i -= 1
            self.values_list.append(self.values)
            self.values = []

    def predict(self, inputs):
        temp_list = []
        for input in inputs:
            # creates a list in [[],[],[]] format for self.values
            for node in range(self.nodes):
                self.values.append([])

            Network.run(dict=input)
            i = self.nodes - 1
            while i > 0:
                Network.feed_forward()
                i -= 1
            temp_list.append(self.values)
            self.values = []
        print(temp_list)


    def stats(self):
        print("Layers :")
        print(self.layers)
        print("Connections :")
        print(self.connections)
        print("Weights :")
        print(self.weights)
        print("Values :")
        print(self.values_list)


Network = Network()

Network.add_layer(8)
Network.add_layer(4)
Network.add_layer(2)

Network.initialize_weights()
#Network.run(dict=[0,1,0,1,0,1,0,0])
#Network.feed_forward()
#Network.feed_forward()
inputs = [[0,1,0,1,0,1,0,0], [0,0,0,1,0,1,0,1]]


Network.train(inputs = [[0,1,0,1,0,1,0,0], [0,0,0,1,0,1,0,1]], labels=[])

Network.stats()


#print(Network.activation_func("Sigmoid", 0))

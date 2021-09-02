
import random
from math import exp

import numpy as np



class Network():

    def __init__(self):
        self.layers = []
        self.connections = []
        self.weights = []
        self.values = []
        self.values_list = []
        self.nodes = 0
        self.learning_rate = 0.1
        self.error_signal_list = []
        self.temp_weight_changes = []


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


    def generate_values(self, inputs):
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


    def generate_error_signal_output(self, label, estimated_output):
        return (label - estimated_output) * estimated_output * (1 - estimated_output)




    def backprop_output(self, labels):
        layers = self.layers
        values_output = [value[len(layers) - 1] for value in self.values_list]
        output_of_previous_neurons = self.values_list
        #[0][len(layers) - 2]
        temp_list = []
        temp_list2 = []
        temp_list_errors = []
        temp_list_errors_storage  = []
        for value_chunk, label_chunk, previous_neurons in zip(values_output, labels, output_of_previous_neurons):
            for value, label in zip(value_chunk, label_chunk):

                error_signal = self.generate_error_signal_output(label, value)
                temp_list_errors.append(error_signal)

                for neuron in previous_neurons[len(layers) - 2]:
                    temp_list2.append(self.learning_rate * error_signal * neuron)
            temp_list.append(temp_list2)
            temp_list2 = []
            temp_list_errors_storage.append(temp_list_errors)
            temp_list_errors = []
        self.error_signal_list.append(temp_list_errors_storage)
        self.temp_weight_changes.append([sum(x) for x in zip(*temp_list)])
        return [sum(x) for x in zip(*temp_list)]

    def generate_error_signal_hidden(self, current_layer = 0):
        preceding_error_list = self.error_signal_list[0]
        single_error_list = []
        error_list = []
        list_error_incl_sample = []
        for value, error_signal in zip(self.values_list, preceding_error_list):
            for error in error_signal:
                for state, weight in zip(value[-2 - current_layer], self.weights[- 1 -current_layer]):
                    single_error_list.append(error * weight * state * (1- state))
                error_list.append(single_error_list)
                single_error_list = []
            list_error_incl_sample.append(error_list)
            error_list = []

        list_error_incl_sample = [[sum(x) for x in zip(*sample)] for sample in list_error_incl_sample]
        self.error_signal_list.reverse()
        self.error_signal_list.append(list_error_incl_sample)
        self.error_signal_list.reverse()
        return list_error_incl_sample


    def backprop_hidden_layer(self, current_layer = 0):
        error_signal_list = self.error_signal_list[0]
        #error_signal_list = self.error_signal_list[0]
        hidden_neuron_list = self.values_list
        # hidden_neuron_list = self.values_list[1 + current_layer]
        sample_list = []
        temp_list = []
        individual_error_signal_list = []
        #print(self.error_signal_list)
        for sample, neuron_values in zip(error_signal_list, hidden_neuron_list):
            for error_signal in sample:
                for neuron_value in neuron_values[-3 - current_layer]:
                    # learning rate * error_signal of hidden layer * value of hidden layer neuron

                    temp_list.append(self.learning_rate * error_signal * neuron_value)
                individual_error_signal_list.append(temp_list)
                temp_list = []

            # Unsure Why different errors
            #individual_error_signal_list =  [sum(x) for x in zip(*individual_error_signal_list)]


            sample_list.append(individual_error_signal_list)
            individual_error_signal_list = []

        np_array = np.array(sample_list)
        length_array = len(np_array[0])
        summed_weights = [[sum(zipped_elements) for zipped_elements in zip(*np_array[:,second_dimension_index])] \
                         for second_dimension_index in range(length_array)]
        flattened_weight_changes = np.array(summed_weights).flatten()
        self.temp_weight_changes.append(flattened_weight_changes.tolist())
        return flattened_weight_changes.tolist()



    def stats(self):
        print("Layers :")
        print(self.layers)
        print("Connections :")
        print(self.connections)
        print("Weights :")
        print(self.weights)
        print("Values :")
        print(self.values_list)

    def run_epoch(self, inputs, labels):
        # Input in form [[1,1,0,1,0], [0,1,1,0,0], [0,1,1,1,0], ..., [0,1,1,1,0]]
        # Need to match input Layer
        self.generate_values(inputs = inputs)
        # Labels in form [[0,1], [1,0], ..., [0,1]]
        # Match Output Layer
        self.backprop_output(labels = labels)

        count_layers = len(self.layers)

        for layer_index in range(count_layers - 2):
            self.generate_error_signal_hidden(current_layer=layer_index)
            self.backprop_hidden_layer(current_layer=layer_index)

        self.temp_weight_changes.reverse()
        self.weights = [np.add(weight_change_layer, weight_layer).tolist() for weight_change_layer, weight_layer in zip(Network.temp_weight_changes, Network.weights)]

Network = Network()

Network.add_layer(8)
Network.add_layer(4)
Network.add_layer(3)
Network.add_layer(2)
Network.initialize_weights()

#Network.run(dict=[0,1,0,1,0,1,0,0])
#Network.feed_forward()
#Network.feed_forward()
inputs = [[0,1,0,1,0,1,0,0], [0,0,0,1,0,1,0,1]]
labels = [[0,1], [0,1]]

Network.run_epoch(inputs, labels)


"""
temp_var = []
for weight_change_layer, weight_layer in zip(Network.temp_weight_changes, Network.weights):
    temp_var.append(np.add(weight_change_layer, weight_layer))

print([len(x) for x in temp_var])


print(Network.weights[2][0], Network.temp_weight_changes[2][0], temp_var[2][0], Network.temp_weight_changes[2][0] + Network.weights[2][0])
print(temp_var[2][0] == Network.temp_weight_changes[2][0] + Network.weights[2][0])
"""

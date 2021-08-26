# tessting numpy
import numpy as np


three_d_list = [[[-0.0006883265978515927, -0.0007217453120373307, -0.0006791937396720466, -0.0006506934142511528], [-0.001013915602236087, -0.0010631418791594927, -0.0010004627625083277, -0.0009584813474311781], [-0.0009642003841555942, -0.0010110129541716986, -0.0009514071761165006, -0.0009114842313906161]], [[-0.0006339509997456869, -0.0006910925571435542, -0.0006830659027122094, -0.0005969061420783826], [-0.0009565433582746606, -0.0010427619733290208, -0.0010306508748552448, -0.0009006478512495039], [-0.0009485510455770621, -0.0010340492686848294, -0.0010220393634139296, -0.000893122568474461]]]

np_array = np.array(three_d_list)

#print(np_array[:,1])


length_index = len(np_array[0])
#for i in range(length_index):
    #print(np_array[:,i])
    #print([sum(x) for x in zip(*np_array[:,i])], "i")
#print([sum(x) for x in zip(*np_array[:,1])])

summed_weights = [[sum(zipped_elements) for zipped_elements in zip(*np_array[:,second_dimension_index])] \
                 for second_dimension_index in range(len(np_array[0]))]
print(summed_weights)
print(np.array(summed_weights).flatten())

"""

[[-0.00101392 -0.00106314 -0.00100046 -0.00095848]
 [-0.00095654 -0.00104276 -0.00103065 -0.00090065]]


"""

import numpy as np

def sigmoid(x):
    """Normalizing function, with this function we normalize the result of the actual neuron

    Args:
        x : Value to normalize

    Returns:
        Normalized value
    """
    return 1 / (1 + np.exp(-x) )

def sigmoid_derivative(x):
    """Function to calculate the error from a given value

    Args:
        x : Value
    
    Returns:
        Error
    """
    return x * (1 - x)

# We want a neural network to predict a value

training_inputs = np.array([  [0, 0, 1],
                              [1, 1, 1],
                              [1, 0, 1],
                              [0, 1, 1],
                              [1, 0, 0],
                              [0, 0, 1],
                              [0, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0],
                              [1, 0, 0],
                              [1, 1, 0] ])

# .T means the transpose array
training_outputs = np.array([ [0,1,1,0,1,0,0,0,0,1,1] ]).T

# Each edge has a weight. We initialize that with a random value
np.random.seed(1)
synaptic_weights = 2 * np.random.random( (3, 1) ) - 1
print('Synaptic weights: ')
print(synaptic_weights)

# Main loop for trainning
for i in range(10000):

    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))


    # Backpropagation, calculating the error
    error = training_outputs - outputs
    adjustments = error * sigmoid_derivative(outputs)

    # Adjusting the weights
    synaptic_weights += np.dot(input_layer.T, adjustments)

print('Outputs after training: ')
for n in outputs:
    print(round(n[0]))

# Ali Ghanabri - 970216657
import numpy as np

def step_activation(x):
    return 1 if x>=0 else 0

def perceptron(x,w,b):
    value = np.dot(x, w) + b
    return step_activation(value)

def xor_logic(x):
    # layer 1 (AND NOT)
    w1 = np.array([2,-1])
    w2 = np.array([-1,2])
    b = -2
    x1 = perceptron(x,w1,b)
    x2 = perceptron(x,w2,b)

    # layer 2 (OR)
    z = np.array([x1, x2])
    w = np.array([2, 2])
    b = -1
    y = perceptron(z,w,b)

    return y

for i in range(2):
    for j in range(2):
        input = np.array([i, j])
        result = xor_logic(input)
        result_test = bool(result)==(bool(i) ^ bool(j))
        print(f'{i} AND {j} = {result} - valid: {result_test}')
# Ali Ghanabri - 970216657
import numpy as np

def step_activation(x):
    return 1 if x>=0 else 0

def perceptron(x,w,b):
    value = np.dot(x, w) + b
    return step_activation(value)

def and_logic(x):
    w = np.array([1,1])
    b = -2
    return perceptron(x,w,b)

for i in range(2):
    for j in range(2):
        input = np.array([i, j])
        result = and_logic(input)
        result_test = bool(result)==(bool(i) and bool(j))
        print(f'{i} AND {j} = {result} - valid: {result_test}')
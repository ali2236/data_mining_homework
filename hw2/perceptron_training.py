# Ali Ghanabri - 970216657
# python 3.10 required
# inputs file generated using MATLAB

from typing import Iterable, Tuple
from nbformat import write
import numpy as np

InputData = Iterable[Tuple[np.array, int]] # (input, expected output)
EpochResult = (int, np.array, np.array) # (error_count, weights, bais)
TrainResult = Tuple[np.array, np.array] # (weights, baises)

def read_data(filename: str) -> InputData:
    data = []
    with open(filename) as inputs:
        for line in inputs.readlines():
            numbers = list(map(int, line.split(',')))
            input = np.array(numbers[:-1])
            output = numbers[-1]
            data.append((input, output))
    return data

def step_activation(x):
    return 1 if x>=0 else 0

def perceptron(x,w,b):
    value = np.dot(x, w) + b
    return step_activation(value)

def epoch(data : InputData, w: np.array, b: np.array, learning_rate: float) -> EpochResult:
    error_count = 0
    for i, d in enumerate(data):
        p = d[0]
        t = d[1]
        a = perceptron(p, w, b[i])
        e = t-a
        if e != 0:
            error_count += 1
            w = w + (e * learning_rate * p)
            b[i] = b[i] + (learning_rate * e)
    return error_count, w, b


def train(data: InputData) -> TrainResult:
    weights = np.random.randint(100, size=(len(data[0][0])))
    baises = np.random.randint(100, size=(len(data)))
    has_errors = True
    print("Training: Errors [weights] [baises]")
    while has_errors:
        result = epoch(data, weights, baises, 0.2)
        error_count, weights, baises = result
        has_errors = error_count > 0
        print(error_count, weights, baises)
    return (weights, baises)

if __name__ == "__main__":
    data = read_data('inputs.csv')
    weights, baises = train(data)
    print('='*110)
    print(f"weights={weights}")
    print(f"bais={baises}")
import argparse
import math
import numpy as np
import csv

np.seterr(all='raise')

def _run(datafile, iterations, alpha):
    # Read CSV file into matrix and split into features and values
    headers, rows = _readcsv(datafile)
    headers.insert(0, 'intercept')  # add the y-intercept as a feature header itself
    matrix = np.matrix(rows)
    features = matrix[:, :-1]
    values = matrix[:, -1]
    features = np.insert(features, 0, 1, axis=1)  # left-pad the features with 1's

    # Run gradient descent
    history = gradientdescent(features, values, iterations, alpha)

    # Get the best parameters from the history
    params = history[-1:, :-1]

    # Print the parameters for the features
    output = ', '.join(['%s = %s' % (key, value) for (key, value) in _mergeheaders(headers, params).items()])
    print(output)

    # Test parameters and print accuracy
    accuracy = testparameters(params, features, values)
    print('Parameters accuracy: %s%%' % round(accuracy, 2))


def _readcsv(file):
    """Read a CSV file into a multidimensional array of rows and columns."""
    rows = []
    with open(file, newline='') as csvfile:
        reader = csv .reader(csvfile, delimiter=',', quotechar='"')
        headers = next(reader, None)  # headers
        for row in reader:
            rows.append([float(x) for x in row])
    return headers, rows


def gradientdescent(features, values, iterations, alpha):
    """Performs gradient descent and returns the parameters associated with their cost for each iteration."""
    m = features.shape[0]  # number of training examples
    n = features.shape[1]  # number of features
    history = np.zeros((iterations, n+1))
    params = np.zeros((n, 1))

    for itr in range(iterations):
        # Perform vectorized gradient descent
        gradient = (1 / m) * features.T * (sigmoid(features * params) - values)
        params = params - alpha * gradient

        # Store the parameters and their associated cost in the history matrix
        history[itr, :-1] = params.T
        history[itr, -1] = cost(features, values, params)

    return history


def cost(features, values, parameters):
    """Computes the cost of applying the parameters to the features."""
    m = features.shape[0]  # number of training examples
    h = sigmoid(features * parameters)
    return (1 / m) * (-values.T * np.log10(h) - (1 - values).T * np.log10(1 - h))


def sigmoid(z):
    """Computes the sigmoid of z, which can be a matrix, vector or scalar."""
    return np.divide(1, np.add(1, np.power(math.e, -z)))


def _mergeheaders(headers, params):
    """Merges the headers from the CSV file with the found parameters into a dictionary."""
    result = {}
    for i, header in enumerate(headers[:-1]):
        result[header] = params.item(i)
    return result


def testparameters(parameters, features, values):
    """Computes the accuracy of the given parameters when applied to the data itself."""
    m = features.shape[0]  # number of training examples
    hits = 0

    for row in range(m):
        test = int(np.asscalar(sigmoid(features[row] * parameters.T)) >= 0.5)
        hits += 1 if test == values[row] else 0

    return (hits / m) * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='the CSV file containing the data')
    parser.add_argument('-a', '--alpha', type=float, default=0.01, help='the learning rate for gradient descent')
    parser.add_argument('-i', '--iterations', type=int, default=1500,
                        help='the number of iterations for gradient descent')
    args = parser.parse_args()
    _run(args.data, args.iterations, args.alpha)

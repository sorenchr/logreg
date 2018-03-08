import argparse
import numpy as np
import csv


def _run(datafile):
    # Read CSV file into matrix and split into features and values
    headers, rows = _readcsv(datafile)
    headers.insert(0, 'intercept')  # add the y-intercept as a feature header itself
    matrix = np.matrix(rows)
    features = matrix[:, :-1]
    values = matrix[:, -1]
    features = np.insert(features, 0, 1, axis=1)  # left-pad the features with 1's
    print(features)


def _readcsv(file):
    """Read a CSV file into a multidimensional array of rows and columns."""
    rows = []
    with open(file, newline='') as csvfile:
        reader = csv .reader(csvfile, delimiter=',', quotechar='"')
        headers = next(reader, None)  # headers
        for row in reader:
            rows.append([float(x) for x in row])
    return headers, rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='the CSV file containing the data')
    args = parser.parse_args()
    _run(args.data)

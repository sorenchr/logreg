import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='the CSV file containing the data')
    args = parser.parse_args()
    print('logreg')
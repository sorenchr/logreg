# logreg

This is a simple implementation of logistic regression using a gradient descent algorithm. The goal is to find the best line that separates a binary classified set of data.

<p align="center">
    <img
      alt="Logistic regression"
      src="logreg.png"
      width="568"
    />
</p>

## Data files

Data files must be in CSV format, with the following structure:

```
x1,x2,y
200,300,32
203,231,42
231,232,13
``` 

Where ``x1,x2,y`` constitutes the header of the data. This project supports multidimensional data, feel free to use any number of features.

## Installation

This project relies on the Python 3 package [NumPy](http://www.numpy.org/). To install the requirements use Python pip: 

```console
$ pip install -r requirements.txt
```

I recommend that you use a [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/) when installing your dependencies.

## Usage

This project can be used from either the terminal, or as an imported module.

### Terminal

```console
$ python logreg.py mydataset.csv
Found the following parameters that best separates the data:
intercept = -3.466760519572871, x1 = 0.13641112675062128
Parameters accuracy: 70.0%
```

The following arguments are available:

- ``-h``,``--help``: Display help on usage
- ``-a``,``--alpha``: Set the learning rate manually (default is 0.01)
- ``-i``,``--iterations``: Set the number of iterations manually (default is 1500)

### Imported module

```python
import logreg
import numpy as np

features = np.asmatrix(np.random.rand(3, 3))
values = np.random.rand(3, 1)

# Gradient descent
iterations = 1500
alpha = 0.01
print(logreg.gradientdescent(features, values, iterations, alpha))

# Cost
parameters = np.random.rand(3, 1)
print(logreg.cost(features, values, parameters))
```

## Requirements

Python 3+ is required, as well as the NumPy package.

## License

Code copyright 2018 Søren Qvist Christensen. Code released under [the MIT license](LICENSE).
# AI from scratch

A neural network for reading handwritten digits (MNIST), written in plain Python. No PyTorch or TensorFlow. Every neuron and every connection is its own Python object, because I wanted to see exactly what a network does at each step instead of calling `model.fit()` and trusting it.

AIMS is just what I named it. Each file in here is a different attempt, and they're kept on purpose so you can see how the idea changed.

## How the network is built

You describe the network as one list of numbers that alternates between neuron layers and connection layers:

```python
Network([784, 1568, 2, 20, 10])
```

That reads as: 784 input neurons (one per pixel of a 28x28 image), then 1568 connections (784 × 2) feeding 2 hidden neurons, then 20 connections (2 × 10) feeding the 10 output neurons, one for each digit. The network figures out how to group the connections from the ratio between layers. The biggest one I tried was `[784, 50176, 64, 2048, 32, 320, 10]`.

Forward propagation walks through that structure neuron by neuron. The network's guess is whichever output neuron has the highest activation, and the error is measured with mean squared error against a one-hot target (so a 6 is `[0,0,0,0,0,0,1,0,0,0]`).

## The versions

| File | What I was trying |
|---|---|
| `AIMS1.0.0(no back propagation).py` | Build the network, run one image through it, measure the error |
| `AIMS1.0.0 Erode.py` | First go at backpropagation ("eroding" the error backwards). It works out how far off each output is, and the plan for the rest is written out as comments |
| `AIMS1.0.0(failed).py` | Skip backprop and train by evolution instead: copy the network 30 times, mutate the copies, keep the best one. Also switched to ReLU and He initialization. Named "failed" because it didn't work |
| `AIMS1.1.0.py` | A full genetic algorithm: 100 networks per generation, split into elites (kept as-is), crossover (children mix two parents' weights) and mutation |
| `MSEtest.py` | Sanity check for the error function |

## What I learned

Evolving tens of thousands of weights at random is a slow way to train a network. Even with crossover and elitism, the population barely moves, and every generation means hundreds of forward passes through Python objects. That's why gradient descent and backprop exist, and that's where this project is headed next.

Building it out of individual objects was also way slower than it needed to be. That was kind of the point, though. Writing every multiply-and-add by hand is how I actually understood what one layer of a network computes.

## Running it

```bash
pip install numpy colorama
python "AIMS1.1.0.py"
```

The scripts load `mnist_train.csv` from a path on my machine, so change the path near the bottom of the file before running. Only the 10,000-image test set is in this repo (`MNIST dataset/mnist_test.csv`). The full training CSV is easy to find online, [this Kaggle version](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) for example. Each row is the label followed by 784 pixel values from 0 to 255.

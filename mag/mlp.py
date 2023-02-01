import random

from backprop import Variable


class Neuron:
    def __init__(self, num_input):
        self._w = [Variable(random.uniform(-1, 1)) for _ in range(num_input)]
        self._b = Variable(random.uniform(-1, 1))

    def __call__(self, x):
        activation = sum((wi*xi for wi, xi in zip(self._w, x)), self._b)
        out = activation.tanh()
        return out

    def parameters(self):
        return self._w + [self._b]


class Layer:
    def __init__(self, num_input, num_output):
        self._neurons = [Neuron(num_input) for _ in range(num_output)]

    def __call__(self, x):
        return [neuron(x) for neuron in self._neurons]

    def parameters(self):
        return [p for neuron in self._neurons for p in neuron.parameters()]


class Mlp:
    def __init__(self, num_in, num_outs):
        size = [num_in] + num_outs
        self._layers = [Layer(size[i], size[i+1]) for i in range(len(size)-1)]

    def __call__(self, x):
        for layer in self._layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self._layers for p in layer.parameters()]


if __name__=="__main__":
    x = [2.0, 3.0, -1.0]
    n = Mlp(3, [4, 4, 1])

    xs = [
        [2.0, -3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1, -1, -1, 1]

    for k in range(10000):
        ypreds = [n(x) for x in xs]

        loss = sum([(ypred-ygt)**2 for ypred, ygt in zip(ypreds, ys)])

        for p in n.parameters():
            p._grad = 0
        loss.backward()

        for p in n.parameters():
            p._val += -0.01 * p._grad

        print(loss._val)

    # loss.draw().render(view=True)

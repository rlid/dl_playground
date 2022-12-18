import math
import numpy as np


class CGraph:
    def __init__(self):
        self.inputs = dict()
        self.consumers = dict()
        self.operations = dict()
        self.u_counter = 0

    def new_var(self):
        u = self.u_counter
        self.u_counter += 1

        self.inputs[u] = []
        self.consumers[u] = []
        return u

    def apply(self, op, *xs):
        y = self.new_var()

        self.inputs[y].extend(xs)
        self.operations[y] = op
        for x in xs:
            self.consumers[x].append(y)

        return y

    def forward(self, xs):
        f_graph = [None for _ in range(self.u_counter)]
        for y in range(self.u_counter):
            op = self.operations.get(y)
            if op:
                f, df = op
                x_values = [f_graph[x] for x in self.inputs[y]]
                f_graph[y] = f(*x_values)
            else:
                f_graph[y] = xs[y]
        return f_graph

    def backward(self, f_graph):
        b_graph = [None for _ in range(self.u_counter)]
        b_graph[-1] = 1
        for y in reversed(range(self.u_counter)):
            xs = self.inputs.get(y)
            if xs:
                _, df = self.operations.get(y)
                x_values = [f_graph[x] for x in self.inputs[y]]
                for j, x in enumerate(xs):
                    if b_graph[x] is None:
                        b_graph[x] = df[j](b_graph[y], *x_values)
                    else:
                        b_graph[x] += df[j](b_graph[y], *x_values)
        return b_graph


cg1 = CGraph()
x1 = cg1.new_var()
y1 = cg1.apply(
    (
        lambda x: x ** 2,
        [lambda dy, x: dy * 2 * x]
    ),
    x1)  # y1 = x^2, dy3/dy1 =
y2 = cg1.apply(
    (
        lambda x: math.exp(x),
        [lambda dy, x: dy * math.exp(x)]
    ),
    y1)  # y2 = exp(y1), dy3/dy2 = -1/y2^2
y3 = cg1.apply(
    (
        lambda x, y: math.log(x) + 1 / y,
        [
            lambda dy, x, y: dy * (1 / x),
            lambda dy, x, y: dy * (- 1 / (y * y))
        ]
    ),
    y1, y2)  # y3 = ln(y1) + 1/y2 = 2*ln(x) + exp(-x^2)

print(x1)
print(y3)
print(cg1.inputs)
print(cg1.consumers)
print(cg1.operations)
print(cg1.u_counter)
fg1 = cg1.forward({x1: 1})
print(fg1)
print(cg1.backward(fg1))

cg2 = CGraph()
w1 = cg2.new_var()
b1 = cg2.new_var()
X1 = np.identity(5)
y1 = cg2.apply(
    (
        lambda w, b: np.matmul(X1, w) + b,
        [
            lambda dy, w, b: np.matmul(X1.T, dy),
            lambda dy, w, b: np.matmul(dy.T, np.ones(dy.shape[0]))
        ]
    ),
    w1, b1
)
z1 = cg2.apply(
    (
        lambda x: np.dot(x, x),
        [lambda dy, x: dy * 2 * x]
    ),
    y1
)
fg2 = cg2.forward(
    {
        w1: np.random.rand(5),
        b1: 1
    }
)

y1_ = np.matmul(X1, fg2[w1]) + fg2[b1]
print(fg2[y1] - y1_)
z1_ = np.dot(y1_, y1_)
print(fg2[z1] - z1_)

bg2 = cg2.backward(fg2)
print(fg2[y1])
print(bg2[y1])

import math


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
                        b_graph[x] = b_graph[y] * df[j](*x_values)
                    else:
                        b_graph[x] += b_graph[y] * df[j](*x_values)
        return b_graph


cg = CGraph()

x0 = cg.new_var()
y1 = cg.apply((lambda x: x ** 2, [lambda x: 2 * x]), x0)  # y1 = x^2, dy3/dy1 =
y2 = cg.apply((lambda x: math.exp(x), [lambda x: math.exp(x)]), y1)  # y2 = exp(y1), dy3/dy2 = -1/y2^2
y3 = cg.apply(
    (
        lambda x, y: math.log(x) + 1 / y,
        [
            lambda x, y: 1 / x,
            lambda x, y: - 1 / (y * y)
        ]
    ), y1, y2)  # y3 = ln(y1) + 1/y2 = 2*ln(x) + exp(-x^2)

print(x0)
print(y3)
print(cg.inputs)
print(cg.consumers)
print(cg.operations)
print(cg.u_counter)
fg = cg.forward({x0: 1})
print(fg)
print(cg.backward(fg))

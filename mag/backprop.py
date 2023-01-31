import math


class Variable:
    def __init__(self, value, label='', _children=(), _op=''):
        self._val = value
        self._grad = 0.0
        self._children = set(_children)
        self._op = _op
        self._label = label

        self._backward = lambda: None
        
    def __repr__(self):
        return f"Var({self._val})"
    
    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self._val+other._val, _children=(self, other), _op='+')

        def _backward():
            self._grad += out._grad
            other._grad += out._grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self._val * other._val, _children=(self, other), _op='*')

        def _backward():
            self._grad += out._grad * other._val
            other._grad += out._grad * self._val

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, power, modulo=None):
        assert isinstance(power, (int, float))
        out = Variable(self._val**power, _children=(self,), label=f'**{power}')

        def _backward():
            self._grad += out._grad * power*(self._val**(power-1))

        out._backward = _backward
        return out

    def tanh(self):
        n = self._val
        t = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)
        out = Variable(t, _children=(self, ), _op='tanh')

        def _backward():
            self._grad += (1 - t**2) * out._grad

        out._backward = _backward
        return out

    def exp(self):
        out = Variable((math.exp(self._val)), _children=(self, ), _op='exp')

        def _backward():
            self._grad += out._val * out._grad

        out._backward = _backward
        return out

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, l):
        self._label = l

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for c in v._children:
                    build_topo(c)
                topo.append(v)

        build_topo(self)

        self._grad = 1
        for node in reversed(topo):
            print(node._val)
            node._backward()

    def draw(self):
        from graphviz import Digraph
        nodes, edges = set(), set()
        
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._children:
                    edges.add((child, v))
                    build(child)

        build(self)

        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
        
        for n in nodes:
            uid = str(id(n))
            dot.node(name=uid, label=f"{n.label}| {n._val} | {n._grad}", shape='record')
            
            if n._op:
                dot.node(name=uid+n._op, label=n._op)
                dot.edge(uid+n._op, uid)

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2))+n2._op)

        return dot


if __name__=='__main__':
    x1 = Variable(2, label='x1')
    x2 = Variable(0, label='x2')

    w1 = Variable(-3, label='w1')
    w2 = Variable(1, label='w2')

    b = Variable(6.881373587019, label='b')

    x1w1 = x1*w1; x1w1.label = 'x1*w1'
    x2w2 = x2*w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1*w1 + x2*w2'
    n = x1w1x2w2 + b; n.label = 'n'

    e = (2*n).exp()
    o = (e-1) / (e+1)
    # o = n.tanh()

    o.backward()

    # a = Variable(2, label='a')
    # b = a + a; b.label='b'
    # c = b/a
    # c.backward()
    # a = Variable(1, "a")
    # b = Variable(2, "b")
    # c = a / b; c.label="z"
    # d = c.exp()
    # c.backward()
    o.draw().render(view=True)




import os
import networkx as nx


INFINITY = float('+inf')

def _from_edges(edges):
    ret = nx.Graph()
    ret.add_edges_from(edges)
    return ret

def PathGraph(n):
    return _from_edges((i, i + 1) for i in range(n - 1))

def CompleteGraph(n):
    return _from_edges((i, j) for i in range(n) for j in range(i))

def ClawGraph(n):
    return _from_edges((n, i) for i in range(n))

def EmptyGraph(n):
    ret = nx.Graph()
    ret.add_nodes_from(range(n))
    return ret

def PetersenGraph():
    return nx.petersen_graph()

if os.name.lower() == 'nt':
    os.system('color')

WHITE   = '\033[0m'
RED     = '\033[31m'
GREEN   = '\033[32m'
BLUE    = '\033[34m'
MAGENTA = '\033[35m'

def _colour(text, colour):
    return f'{colour}{text}{WHITE}'

def red(text):
    return _colour(text, RED)

def green(text):
    return _colour(text, GREEN)

def blue(text):
    return _colour(text, BLUE)

def magenta(text):
    return _colour(text, MAGENTA)


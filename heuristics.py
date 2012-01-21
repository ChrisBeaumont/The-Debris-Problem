from network import mst, dijkstra
from solvers import greedy_search
from random import choice
import numpy as np

def h1(solution, option):
    """ Heuristic that measures the total unmet demand of a solution + option """
    satisfied = solution.satisfied_vertices
    satisfied.add(option.v1)
    satisfied.add(option.v2)
    return solution.total_demand - sum([v.demand for v in satisfied])

def h2(solution, option):
    """ Measures unmet demand for each node, weighted by
    the minimum possible path weight to that node """
    s = solution.step(option)
    v = s.vertices
    e = s.edges
    c = set(s.decisions)
    [cc.clear() for cc in c]
    
    #XXX generalize to many supplies
    paths = dijkstra(v, e, s.supplies[0])
    result = sum([paths[i] * v[i].demand for i in range(len(v))])

    [cc.restore() for cc in c]
    
    return result

def h3(solution, option):
    """ Measures weight of MST, ignoring weight of already-cleared roads """
    s = solution.step(option)
    v = s.vertices
    e = s.edges
    c = s.decisions

    [cc.clear() for cc in c]
    e_mst = mst(v, e)
    result = sum([e.weight for e in e_mst])
    [cc.restore() for cc in c]

    return result

def h4(solution, option):
    """ Combined weight of shortest path between supply and node * demand of node """
    s = solution.step(option)
    v = s.vertices
    e = s.edges
    c = s.decisions

    #XXX generalize to many supplies
    [cc.clear() for cc in c]
    e_mst = mst(v, e)
    paths = dijkstra(v, e_mst, s.supplies[0])
    result = sum([v[i].demand * paths[i] for i in range(len(v))])
    [cc.restore() for cc in c]

    return result

def h5(solution, option):
    """ Cost of solving rest of problem using greedy search w/ h1"""
    best = solution.step(option)
    best._heuristic_func = h1
    best = greedy_search(best)
    return best.objective()

def h7(solution, option, ntrial = 100):
    """ Best solution obtained by random walking from current path to solution """
    start = solution.step(option)
    best = np.inf
    for i in xrange(ntrial):
        s = start
        while not s.satisfied():
            s = s.step(choice(list(s.options)))
        val = s.objective()
        if val < best:
            best = val
    return best

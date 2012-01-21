from random import choice
from bisect import bisect_left
from random import choice
import numpy as np

from network import min_path
np.seterr('ignore')

"""
A variety of functions to solve the debris problem
"""

def greedy_search(solution, verbose=False):
    """ Use a solution's heuristic function to solve the problem via greedy search """
    while not solution.satisfied():
        os = list(solution.options)
        if len(os) == 0:
            solution.plot()
            raise Exception("Could not find a solution!")

        f = np.array([solution.heuristic(o) for o in os])
        best = f.argmin()
        if verbose:
            print "Choosing option %s with heuristic value of %f" % (os[best], f[best])
        solution = solution.step(os[best])

    if verbose:
        print "Solved at a total penalty of %f" % solution.objective()
    return solution


def depth_limited_search(solution, depth):
    """ Internal method used by greedy_recursion """
    small = -100000 * (depth + 1)
    if solution.satisfied():
        return None, small

    os = list(solution.options)
    if len(os) == 0:
        raise Exception("Could not find a solution!")

    if depth <= 0:
        f = np.array([solution.heuristic(o) for o in os])
        best = f.argmin()
        return os[best], f[best]

    result, best = None, np.inf
    for o in os:
        r,b = depth_limited_search(solution.step(o), depth-1)
        if b < best:
            result, best = o,b
    return result, best
                        

def greedy_recursion(solution, depth = 5, verbose = False):
    """ Use greedy search + partial recursion (depth levels deep in the
    recursion tree) to complete a given solution """
    while not solution.satisfied():
        step, value = depth_limited_search(solution, depth)
        solution = solution.step(step)
        if verbose:
            print "Choosing option %s with hueristic of %f" % (step, value)

    if verbose:
        print "Solved at a total penalty of %f" % solution.objective()
    return solution

def choice_prob(options, probs):
    """ Chose among a set of options, with probability of selection
    propritional to probs

    Inputs:
    =======
    options : list of options
    probs : list of corresponding probability

    Outputs:
    ========
    One of the input options
    """

    cdf = np.cumsum(probs)
    tot = cdf[-1]
    rand = np.random.uniform(high=tot)
    choice = bisect_left(cdf, rand)
    return options[choice]    

def greedy_mc_search(solution):
    """ Non-deterministic greedy search,
    where probability of choosing a given option is proportional to 
    its heuristic value """
    while not solution.satisfied():
        os = list(solution.options)
        f = [solution.heuristic(o) for o in os]
        solution = solution.step(choice_prob(os, f))
    return solution
        
def random_search(solution):
    """ Random search to a solution """
    while not solution.satisfied():
        o = choice(list(solution.options))
        solution = solution.step(o)
    return solution

def random_search_2(solution):
    """ Random search to a solution. More informed than random_search.
    While not solved, the method chooses an unsatisfied vertex, with
    probability proportional to its demand. It then clears a route
    from the supply center to that vertex, and repeats.
    """
    vs = solution.vertices
    es = solution.edges
    while not solution.satisfied():
        #pick an unsatisfied node, and go there
        todo = list(solution.unsatisfied_vertices)
        pri = [t.demand for t in todo]
        if len(todo) != 0:
            target = choice_prob(todo, pri)
            source = choice(solution.supplies)
            [e.clear() for e in solution.decisions]
            path = min_path(vs, es, source, target)
            [e.restore() for e in solution.decisions]
            path = filter(lambda x: x not in solution.decisions, path)
            solution = solution.step_many(path)
        else:
            print 'defaulting to greedy'
            return greedy_search(solution)
    return solution

def random_many(solution, ntrial=1000):
    """ Execute many random solutions, and return the best """
    start = solution
    best, result = np.inf, None
    for i in xrange(ntrial):
        s = random_search(start)
        val = s.objective()
        if val < best:
            best, result = val, s

    return result

def anneal(solution, cooling = None, ntrial = 10000, verbose=False):
    """ Use simulated annealing to improve upon a sub-otpimal solution.

    Inputs:
    =======
    Solution: Sub-optimal solution
    cooling : function (optional)
      The cooling schedule. cooling(i) gives the temperature at step i.
      defaults to a constant temperature of 1e-5
    ntrial : int
      Number of iterations to anneal for. Defaults to 10,000

    Outputs:
    ========
    An iterator object. Each call to iterator.next() iterates until
    annealing is finished, or until a new best solution is found. 
    It returns the best solution everytime it is found
    """

    s, v = solution, solution.objective()
    best, bestv = s, v
    for i in xrange(ntrial):
        s2 = s.nudge()
        v2 = s2.objective()
        t = cooling(i) if cooling is not None else 1e-5
        de = v2 - v
        if t == 0 and de > 0: continue
        prob = 0 if de / t > 100 else np.exp(-1 * de / t)
        if np.random.uniform() < prob:
            if verbose: 
                print v2, t
            s, v = s2, v2
            if v < bestv:
                best, bestv = s, v
                yield best, bestv


def dynamic_anneal(solution, cooling = None, ntrial = 10000, verbose=False):
    """ An attempt to dynamically control the temperature during annealing. Not
    well developed yet """
    s, v = solution, solution.objective()
    best, bestv = s, v
    hist_len = ntrial / 100
    history = np.zeros(hist_len)
    t = cooling(0) if cooling is not None else 1e-5

    for i in xrange(ntrial):
        s2 = s.nudge()
        v2 = s2.objective()
        de = v2 - v
        history[i % hist_len] = de

        #if i % hist_len == 0 and i > 0:
        #    t = max(0, t - 0.2 * history.std())
                    
        if t == 0 and de > 0: continue
        prob = 0 if de > t * 100 else np.exp(-1 * de / t)
        if np.random.uniform() < prob:
            if verbose: 
                print v2, t
            s, v = s2, v2
            if v < bestv:
                best, bestv = s, v
                yield best, bestv

def cool_exp(x, start, stop, niter):
    """ An exponential cooling function, which decreases from start to
    stop in niter iterations"""
    logf = np.log(1. * stop / start) / niter
    logt = np.log(start) + x * logf
    return np.exp(logt)

def cool_lin(x, start, stop, niter):
    """ A linear cooling function, which decreases from start to stop in
    niter iterations"""
    return start + 1. * (stop - start) * x / niter

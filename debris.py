import numpy as np
from random import randint, choice

from network import *
import heuristics as hur
from solvers import choice_prob

class Solution(object):

    def __init__(self, vertices, edges, resources, supplies, 
                 total_demand = None, 
                 options=None, 
                 cleared=None, 
                 decisions = None,
                 heuristic = hur.h1, keep=False):
        """ Create a new solution object

        Input:
        ======
        vertices : list
          List of vertices in the graph
        edges : list
          List of edges in the graph
        resources : list
          List of resources available on each day
        supplies : list
          references to vertices which are supply shelters
        keep : boolean (default False)
          If True, the decision lists will be used as is. If False, 
          it will be trimmed of useless steps
        total_demand : float (optional)
          Total demand in the graph. Will be calculated if not provided
        options : set (optional)
          Possible options to make on the next step. Will be calculated
          from initial supply centers if not provided
        cleared : list (optional)
          List of cleared sights. Will be calculated if not supplied
        heuristic : function (H1 default)
          Heuristic to rate the various options          
        """

        self._vertices = vertices
        self._edges = edges
        self._resources = resources
        self._supplies = supplies
        if len(supplies) > 2:
            raise Exception("Only set up to handle 1 or 2 supply depots")

        # These attributes are accessed via property functions.
        # In some cases, they are only calculated when requested.
        # this saves some computation time
        self._heuristic_func = heuristic
        self._decisions = []        
        self._total_demand = None
        self._options = None
        self._cleared = None
        self._supply_connections = None
        self._connect_step = None
        self._satisfied_vertices = None

        assert sum([s.supply for s in supplies]) >= sum([v.demand for v in vertices]), \
            "Not enough supply to meet all demand"
        
        if total_demand is not None:
            self._total_demand = total_demand

        if options is not None:
            self._options = options

        if cleared is not None:
            self._cleared = cleared

        if decisions is not None:
            self._decisions = decisions
            self.refine_path(keep=keep)
            

    @property
    def vertices(self):
        """ A list of vertices in the graph """
        return self._vertices

    @property
    def edges(self):
        """ A list of edges of the graph """
        return self._edges

    @property
    def resources(self):
        """ A list of resource amounts for each period """
        return self._resources

    @property
    def supplies(self):
        """ A list of the supply vertices """
        return self._supplies

    @property
    def total_demand(self):
        """ The total demand to be satisfied on the graph """
        if self._total_demand is None:
            self._total_demand = sum([abs(v.demand) for v in self.vertices])
        return self._total_demand

    @property
    def options(self):
        """ A set of the edges that can be cleared from the current state """
        if self._options is None:
            ss = self.supplies
            cleared = self.cleared
            sconn = self._supply_connections
            options = set([e for v in cleared for e in v.edges if e not in self._decisions])
            
            for o in list(options):
                if (o.v1 not in cleared) or (o.v2 not in cleared): continue
                if self._connect_step is not None:
                    options.remove(o)
                else:
                    if (o.v1 in sconn[ss[0]] or o.v2 in sconn[ss[0]]) and \
                            (o.v1 in sconn[ss[1]] or o.v2 in sconn[ss[1]]):
                        pass
                    else:
                        options.remove(o)
            self._options = options
        
        return self._options

    @property
    def decisions(self):
        """ The history of which edges were cleared """
        return self._decisions

    @property
    def cleared(self):
        """ A set of the vertices that have been cleared """

        if self._cleared is None:
            self._cleared = set(self.supplies)
            for d in self.decisions:
                self._cleared.add(d.v1)
                self._cleared.add(d.v2)
        return self._cleared

    @property 
    def satisfied_demand(self):
        """ The amount of demand that has been satisfied so far """
        vs = self.satisfied_vertices
        if len(self.supplies) == 1 or self._connect_step is not None:
            return sum([v.demand for v in vs])
        if len(self.decisions) == 0: return False

        result = 0
        for s in self.supplies:
            dem = sum([v.demand for v in self._supply_connections[s]])
            dem = min(dem, s.supply)
            result += dem
        return result

    @property
    def satisfied_vertices(self):
        """ A set of the vertices that have been satisfied so far """
        if self._satisfied_vertices is None:
            self._satisfied_vertices = set([v for e in self.decisions for v in [e.v1, e.v2]])
        return self._satisfied_vertices

    @property
    def cleared_edges(self):
        """ A set of edges that have been cleared so far """
        return self.decisions

    @property
    def unsatisfied_vertices(self):
        """ A set of vertices that have not been satisfied so far """
        satisfied = self.satisfied_vertices
        all = set(self.vertices)
        return all.difference(satisfied)

    def min_possible_cost(self):
        """ Find a lower bound on the objective function, assuming nodes are
        visited in the ideal order, along the cheapest edge """
        vs = sorted(self.vertices, key = lambda x: -1 * x.demand)
        es = [sorted(v.edges, key = lambda x: x.weight)[0] 
              for v in vs]
        unmet_demand = self.total_demand
        result, day_cost, day = 0, 0, 0
        rs = self.resources
        for v,e in zip(vs, es):
            delta = rs[min(day, len(rs)-1)] - day_cost - e.weight
            if day_cost + e.weight > rs[min(day, len(rs)-1)]:
                result += unmet_demand
                day += 1
                day_cost = e.weight
            else:
                day_cost += e.weight
            unmet_demand -= v.demand

        return result

    def refine_path(self, keep = False):
        """ Removes decisions which neither explore new
        territory nor connect the two supply depots """

        sconn = defaultdict(set)
        all_connect = False
        [sconn[s].add(s) for s in self.supplies]
        cleared = set()
        [cleared.add(s) for s in self.supplies]
        remove = set()

        for i,d in enumerate(self._decisions):

            c = False
            if not all_connect:
                c = True             
                for s in self.supplies:
                    if (d.v1 in sconn[s]) or (d.v2 in sconn[s]):
                        sconn[s].add(d.v1)
                        sconn[s].add(d.v2)
                    else:
                        c = False
                if c:
                    self._connect_step = d
                    all_connect = True

            if (d.v1 in cleared) and (d.v2 in cleared) and (not c):
                remove.add(i)
                
            if (d.v1 not in cleared) and (d.v2 not in cleared):
                self.plot()
                raise Exception("Illegal Move")

            cleared.add(d.v1)
            cleared.add(d.v2)
    
        self._supply_connections = sconn            
        if not keep:
            self._decisions = [self._decisions[i] for i in range(len(self._decisions))
                               if i not in remove]

        self._cleared = cleared

    def nudge(self):
        """ Create a modified version of the current solution. The output
        decision list consists of 3 parts: 
        
        begin: The first N steps of the current solution. N chosen at
        random

        middle: The shortest-weight between a random supply depot and
        vertex not visited in begin

        end: The rest of the current solution, trimmed of work
        duplicated during middle
        
        Outputs:
        ========
        A new solution instance
        """

        d = self.decisions

        # truncate the solution at some point
        pos = randint(0, len(d)-1)        
        begin, end = d[:pos], d[pos:]
        
        # choose a random vertex not satisfied before steps in beginning
        # connect it to a random source node
        # probability of choosing a vertex proportional to its demand
        satisfied = set([v for e in begin for v in [e.v1, e.v2]])
        [satisfied.add(s) for s in self.supplies]

        unsatisfied = list(set(self.vertices).difference(satisfied))
        pris = [v.demand for v in unsatisfied]
        target = choice_prob(unsatisfied, pris)
        source = choice(self.supplies)

        # find shortest path from supply node to target
        [e.clear() for e in begin]
        middle = min_path(self.vertices, self.edges, source, target)
        [e.restore() for e in begin]
        bset = set(begin)
        middle = filter(lambda x: x not in bset, middle)
        
        # add the rest of original solution
        # init method will remove redundant work
        path = begin + middle + end
        result = Solution(self.vertices, self.edges, self.resources, self.supplies, 
                          heuristic = self._heuristic_func, decisions = path)
        return result

    def step(self, decision, keep=False):
        """ Create and return a new Solution instance, generated by combining
        the current solution with a new step.

        Inputs:
        =======
        decision : Edge object
           A valid edge to clear

        Outputs:
        ========
        Solution instance
        A new solution
        """
        assert decision not in self.decisions, "Duplicate decision %s" % decision
        assert (decision.v1 in self.cleared) + (decision.v2 in self.cleared) != 0
        assert decision in self.options

        cleared = set(self.cleared)
        cleared.add(decision.v1)
        cleared.add(decision.v2)
        result = Solution(self.vertices, self.edges, self.resources, self.supplies, 
                          total_demand = self.total_demand,
                          cleared=cleared, heuristic = self._heuristic_func, 
                          decisions = self.decisions + [decision], keep=keep)
        return result

    def step_many(self, options):
        """ Take several steps at once, and return the result
        Inputs:
        =======
        options : List of edges
          The edges to clear, in order
          
        Outputs:
        ========
        A new solution instance, obtained by executing each step in options
        """
        result = self
        for o in options:
            result = result.step(o)
        return result
            
    def split_by_days(self):
        """
        Split the solution's decisions by days, based on the amount
        of work available in each day. 

        Outputs:
        ========
        A list of lists of decisions. Result[i] lists the decisions executed on day i
        """

        unmet_demand = self.total_demand
        day = 0
        day_cost = 0
        result = []
        current = []
        rs = self.resources
        for d in self.decisions:
            if day_cost + d.weight > rs[min(day, len(rs)-1)]:
                day_cost = d.weight
                day += 1
                result.append(current)
                current = [d]
            else:
                current.append(d)
                day_cost += d.weight

        result.append(current)
        return result

    def objective(self, satisfied_history = None):
        """ Evaluates the objective function (sum of unmet demand after every
        period) for a complete, feasible solution 

        Parameters:
        ===========
        satisfied_history : empty list (optional)
          On output, will contain the amount of demand satisfied on each day
        """

        #XXX this will work for 2 supply depots, as long as 
        #they have enough resources to supply all demand.
        #It will correctly track one hospital running
        #out of supply

        unmet_demand = self.total_demand
        result = 0

        sconn = self._supply_connections
        supply = {}
        cleared = set()
        ss = self.supplies
        for s in ss:
            supply[s] = s.supply

        all_connect = False
        leftover_demand = 0
        if satisfied_history is None: satisfied_history = []
        else: satisfied_history.clear()

        day = -1
        rs = self.resources
        for decisions in self.split_by_days():
            assert unmet_demand >= 0
            day_cost = 0
            day += 1
            for d in decisions:
                day_cost += d.weight
                assert day_cost < self.resources[min(day, len(rs)-1)]
                if all_connect or d == self._connect_step:
                    unmet_demand -= leftover_demand
                    leftover_demand = 0
                    all_connect = True
                for v in [d.v1, d.v2]:
                    if v in cleared: continue
                    cleared.add(v)
                    if all_connect:
                        unmet_demand -= v.demand                        
                    else:
                        dem = v.demand
                        for s in ss:
                            if v not in sconn[s]: continue
                            sub = min(dem, supply[s])
                            dem -= sub
                            unmet_demand -= sub
                            supply[s] -= sub
                        leftover_demand += dem
            satisfied_history.append(self.total_demand - unmet_demand)
            result += unmet_demand
            
        if unmet_demand > 1e-3: 
            return np.inf

        assert unmet_demand < 1e-3, unmet_demand
        assert leftover_demand < 1e-3, leftover_demand
        return result
                
    def feasible(self):
        """ Determine if a solution is feasible """
        raise NotImplementedError()

    def heuristic(self, option):
        """ Evaluate the heuristic value of a given (perhaps partial) solution""" 
        return self._heuristic_func(self, option)
    
    def satisfied(self):
        """ Returns true if all demand has been satisfied """
        return np.isfinite(self.objective())

    def __str__(self):
        """ Describe the solution """
        result = ''
        satisfied = []
        obj = self.objective(satisfied)
        for i, day in enumerate(self.split_by_days()):
            labor = sum([d.weight for d in day])
            possible_labor = self.resources[min(i, len(self.resources)-1)]
            result += ('Day %i labor: %0.1f/%0.1f, satisfied:%0.1f):\n\t%s\n' % 
                       (i, labor, possible_labor, satisfied[i], '\n\t'.join([str(d) for d in day])))
        result += '\nPenalty: %i' % self.objective()
        return result

    def plot(self, out=None):
        """ Plot the solution 
        Parameters
        ==========
        out : string (optional)
          If present, save the figure to file. Otherwise, display
        """
        plot_graph(self.vertices, self.edges, self, out=out)

    def official_output(self):
        """
        Returns the string representing the solution in the format
        specified by the competition
        """
        days = {}
        result = []
        rs = self.resources
        for i, ds in enumerate(self.split_by_days()):
            day_cost = 0
            for d in ds:
                days[d] = i
                day_cost += d.weight
            assert day_cost < rs[min(i, len(rs)-1)]

        for e in self.edges:
            x = e.v1.id
            y = e.v2.id
            x,y = min(x,y), max(x,y)
            key = (x,y)
            z = days[e] if e in days else -1
            result.append("%i %i %i\n" % (x, y, z))
        return ''.join(result)
                          
    def write(self, filename):
        """ Write the solution to filename in the official contest format """
        f = open(filename, 'w')
        f.write(self.official_output())
        f.close()

def read(file):
    """ Read a problem, contained in the files file_nodes.txt,
    file_edges.txt and file_resources.txt. 

    Input:
      file : string
      the prefix of the input files

    Output:
      A triplet of lists, containing vertices, edges, and resources
    """
    vs = []
    es = []
    rs = []
    used = set()
    for i,d in enumerate(open(file+'_nodes.txt').readlines()):
        if len(d.strip()) == 0: continue
        id, supply, x, y = d.strip().split()
        vs.append(Vertex(int(id), float(x), float(y), float(supply)))
        
    used = set()
    for d in open(file+'_edges.txt').readlines():
        if len(d.strip()) == 0: continue
        v1, v2, cost = d.strip().split()
        v1 = int(v1)
        v2 = int(v2)
        key = (min(v1, v2), max(v1, v2))
        if key in used: continue
        assert key not in used, key
        used.add(key)
        cost = float(cost)
        assert vs[v1].id == v1
        assert vs[v2].id == v2
        es.append(Edge(vs[v1], vs[v2], cost))

    for d in open(file+'_resources.txt').readlines():
        if len(d.strip()) == 0: continue
        day, resource = map(int, d.strip().split())
        rs.append(resource)

    return vs, es, rs
                  
                      
def plot_graph(vs, es, solution = None, out = None):
    """ Plot the graph """
    import matplotlib.pyplot as plt
    import matplotlib

    x = np.array([v.lon for v in vs])
    y = np.array([v.lat for v in vs])
    z = np.array([v.supply for v in vs])
    fig = plt.figure(figsize = (15,10))
    sub = fig.add_subplot(111, aspect='equal')

    # plot edges
    thick = np.array([e.weight for e in es])
    for e in es:
        xx = [e.v1.lon, e.v2.lon]
        yy = [e.v1.lat, e.v2.lat]
        tt = e.weight
        plt.plot(xx,yy, 'k-', linewidth = tt / max(thick) * 2, alpha = 0.2)

    #colorbar
        cdict = {'red': ((0.0, 0.0, 0.0),
                         (0.5, 0.0, 0.0),
                         (1.0, 1.0, 1.0)),
                 'green': ((0.0, 0.0, 0.0),
                           (0.0, 0.0, 0.0),
                           (1.0, 0.0, 0.0)),
                 'blue': ((0.0, 0.0, 0.0),
                          (0.5, 0.5, 0.5),
                          (1.0, 0.5, 0.5))}
        my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
        my_cmap = 'hot'

    # plot demand points
    good = (z <= 0)
    c = np.abs(z[good])
    p = plt.scatter(x[good], y[good], c=c, 
                    cmap=my_cmap, s = 30, vmin = c.min(), vmax = c.max(), 
                    alpha = 1.0, edgecolor='grey', zorder=1000)

    cbar = plt.colorbar()
    cbar.set_label('Demand')
    
    #plot resource points
    good = z > 0
    sz_bg = 800.
    s = z[good]
    s = 1. * s  / s.max() * sz_bg
    plt.scatter(x[good], y[good], s = s, c = 'b', edgecolor='none', 
                marker='d', zorder = 5000)

    if solution is not None:
        plot_solution(solution)
    if out is not None:
        plt.savefig(out)
    else:
        plt.show()


def plot_solution(s):
    """ Overplot the solution, after a previous call to plot_graph. """
    import matplotlib.pyplot as plt
    colors = ['#000000', '#081D58', '#253494', '#225EA8', '#1D91C0', '#41B6C4', 
              '#7FCDBB', '#C7E984', '#EDF881']
    for j,ds in enumerate(s.split_by_days()):
        color = colors[min(j, len(colors)-1)]
        for i,e in enumerate(ds):
            x = [e.v1.lon, e.v2.lon]
            y = [e.v1.lat, e.v2.lat]
            plt.plot(x, y, linewidth = 4, zorder=1, color = color)
        #plt.annotate('%i' % i, xy = (sum(x)/2, sum(y)/2), xytext = (sum(x)/2, sum(y)/2))
        
def read_solution(problem, solution):
    """ Read a solution file, and import into a new solution object:
    NOTE:
    The solution object may be slightly different than the input file, due to the
    ambiguity of the order that actions within a given day are performed. This may affect
    the objective function value

    Input:
    Problem: string
        The file prefix for the problem files that the solution addresses
    solution: string
        The filename of the solution
    """
    v, e, r = read(problem)
    supplies = [vv for vv in v if vv.supply > 0]
    data = open(solution).readlines()
    s = Solution(v, e, r, supplies)
    
    data = [map(int, d.strip().split()) for d in data]
    data = sorted(data, key = lambda x: x[2])
    nday = data[-1][2]
    nuse = 0
    ntot = len([d for d in data if d[2] != -1])
    decisions = []
    cleared = set()
    [cleared.add(s) for s in supplies]
    
    for day in xrange(nday+1):
        candidates = [d for d in data if d[2] == day]
        nuse += len(candidates)
        while len(candidates) != 0:
            for c in list(candidates):
                assert c[2] == day
                step = [ee for ee in e if ee.v1.id == c[0] and 
                        ee.v2.id == c[1]]
                if len(step) != 1:
                    print ' '.join("%s" % e for e in step)
                assert len(step) == 1
                step = step[0]
                assert step.v1.id == c[0] and step.v2.id == c[1]
                if step.v1 in cleared or step.v2 in cleared:
                    cleared.add(step.v1)
                    cleared.add(step.v2)
                    decisions.append(step)
                    candidates.remove(c)

    #lots of sanity checks here
    assert len(decisions) == ntot, "%i, %i, %i" % \
        (len(decisions), ntot, nuse)
    backup = list(decisions)
    assert decisions is not backup
    result = Solution(v, e, r, supplies, decisions = decisions, keep=True)
    assert [x == y for x,y in zip(backup, result.decisions)]

    for i,ds in enumerate(result.split_by_days()):
        for dd in ds:
            hit = [d for d in data if d[0] == dd.v1.id and d[1] == dd.v2.id]
            if len(hit) != 1:
                for h in hit:
                    print h
            assert len(hit) == 1, len(hit)
            hit = hit[0]
            #if hit[2] != i: print hit[2], i            

    return result

if __name__ == "__main__":
    from solvers import *
    import matplotlib.pyplot as plt

    problem = 'AICS_CH'
    v, e, r = read(problem)
    supplies = [vv for vv in v if vv.supply > 0]
    s = Solution(v, e, r, supplies)
    s = random_solution_2(s)
    print s.objective()
    

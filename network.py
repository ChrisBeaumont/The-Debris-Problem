import numpy as np
from collections import defaultdict
from heapq import heapify, heappop
from pq import pq

class Edge(object):
    """ Represents an edge in the debris problem

    Attributes:
    ===========
    v1: Vertex object
       Start of edge (edges are undirected)
    v2 : Vertex object
       End of edge (edges are undirected)
    weight : Amount of debris on current edge

    Side Effects
    ============
    Current edge will be appended to each of v1 and v2's edges lists
    """
    def __init__(self, v1, v2, weight):
        self.v1 = v1
        self.v2 = v2
        self.weight = weight
        self._backup_weight = weight
        self.cleared = False
        self.v1.edges.append(self)
        self.v2.edges.append(self)

    def clear(self):
        """ Zero out (but remember) the weight on this node """
        self.weight = 0

    def restore(self):
        """ Undo the effect of clear(), restoring original weight """
        self.weight = self._backup_weight

    def __lt__(self, other):
        """ Nodes are ranked by weight """
        return self.weight < other.weight

    def __str__(self):
        return '(%i,%i,%0.1f)' % (self.v1.id, self.v2.id, self.weight)
    
class Vertex(object):
    """ Represents a vertex object in the debris problem

    Attributes:
    ===========
    id: Integer id
    lon : float
        x coordinate
    lat : float
        y coordinate
    supply : float
        Amount of supply associated with this node. 0 if demand center
    demand : float
        Amount of demand at this vertex. 0 if a supply center
    edges:
        The edges connected to this vertex. List of edge objects
    """
    def __init__(self, id, lon, lat, supply):
        self.id = id
        self.lat = lat
        self.lon = lon
        self.supply = supply
        self.demand = max(0, -1 * supply)
        self.edges = []

    def add_edge(self, edge):
        """ Add an edge to this vertex. Called automatically by Edge.__init__ """
        assert edge not in self.edges
        self.edges.append(edge)

    @property
    def adjacent(self):
        """ List of vertices connected to this vertex by an edge """
        result = set([v for e in self.edges for v in [e.v1, e.v2]])
        result.remove(self)
        return result

    def __lt__(self, other):
        """ Used by dijkstras algorithm to order vertices """
        return self._ss_d < other._ss_d
        
def mst(v, e):
    """ Compute the MST, using Prims algorithm 

    Inputs:
    ======
    v: List of vertex objects, describing the graph
    e: List of edge objects, describing the graph

    Outputs:
    ========
    A new set of edges, describing the minimum spanning tree

    """
    vold = set(v)
    vnew = set([v[0]])
    result = set()
    edges = sorted(e)
    
    while vnew != vold:
        for uv in edges:
            old = (uv.v1 in vnew) + (uv.v2 in vnew)
            if old == 1:
                vnew.add(uv.v1)
                vnew.add(uv.v2)
                result.add(uv)
                #edges.remove(uv)
                break
            
    return result
    

def initialize_single_source(vs, es, s):
    """ Setup to Dijkstras algorithm """
    for v in vs:
        v._ss_d = np.inf
        v._ss_pi = None
        v._ss_edge = None
    s._ss_d = 0

def relax(u, v, e, Q):
    w = e.weight
    if v._ss_d > u._ss_d + w:
        v._ss_d = u._ss_d + w
        v._ss_pi = u
        v._ss_edge = e
        Q.add(v)

def dijkstra(vs, es, s, stop = None):
    """ Dijkstra's algorithm to compute the shortest distance to all
    vertices vs from a given source vertex s by traveling the edges e
    """
    initialize_single_source(vs, es, s)
    key = lambda x: -1 * x._ss_d
    Q = pq(vs)
    edict = defaultdict(set)
    for e in es:
        edict[e.v1].add(e)
        edict[e.v2].add(e)

    for i in range(len(vs)):
        # min path to u is determined at end of loop
        u = Q.pop()
        for e in edict[u]:
            v = e.v1 if e.v1 != u else e.v2
            relax(u, v, e, Q)
        if u == stop: return        
        
    result = [v._ss_d for v in vs]
    return result

def min_path(vs, es, source, target):
    """ Return min path from source to target, on the graph 
    described the vertices vs and edges es

    Parameters:
    ===========
    vs: List of vertex objects
    es: List of edge objects
    source : Starting vertex
    target : Ending vertex

    Outputs:
    ========
    List of edges, connecting source to target along the minimum-weight path
    """
    dijkstra(vs, es, source, stop = target)
    test = target
    result = []
    while test != source:
        e = test._ss_edge
        result.append(e)
        test = e.v1 if e.v1 != test else e.v2
    assert test == source and test._ss_edge is None
    return result[::-1]

def is_spanning(vs, es):
    """ Tests whether the graph described by vertices vs and edges es is a
    spanning tree """
    [e.clear() for e in es]
    d = dijkstra(vs, es, vs[0])
    [e.restore() for e in es]
    return max(d) < 1e-5

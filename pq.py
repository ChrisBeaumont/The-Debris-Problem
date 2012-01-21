import itertools
from heapq import heappush, heappop, heapify

class pq(object):
     """ A priority queue object, taken from the Python website and adapted for
     use with debris vertices """
    def __init__(self, objects):
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = itertools.count()
        self.pq = []
        [self.add(o) for o in objects]
        
    def add(self, task):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove(task)
        priority = task._ss_d
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')

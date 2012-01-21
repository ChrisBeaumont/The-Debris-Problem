from debris import *
from solvers import *
from network import min_path
from random import choice

v,e,r = read('cambridge')
supplies = [vv for vv in v if vv.supply > 0]
s = Solution(v, e, r, supplies)

s = greedy_search(s)
s.plot(out='nudge1.png')

d = s.decisions
truncate = 40
begin = d[:truncate]
satisfied = set([vv for ee in begin for vv in [ee.v1, ee.v2]])
unsatisfied = list(set(v).difference(satisfied))
target = choice(unsatisfied)

[ee.clear() for ee in begin]
middle = min_path(v, e, supplies[1], target)
[ee.restore() for ee in begin]
end = d[truncate:]

s2 = Solution(v, e, r, supplies, decisions = begin)
s2.plot(out='nudge2.png')

s3 = Solution(v, e, r, supplies, decisions = begin + middle)
s3.plot(out='nudge3.png')

s4 = Solution(v, e, r, supplies, decisions = begin + middle + end)
s4.plot(out='nudge4.png')

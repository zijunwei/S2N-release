
import sys
import cplex
from cplex.exceptions import CplexError


def isOverlap(seg_a, seg_b):
    left = min(seg_a[0], seg_b[0])
    right = max(seg_a[1], seg_b[1])
    union_szie = right-left
    sep_size = seg_a[1] - seg_a[0] + seg_b[1] - seg_b[0]
    if union_szie < sep_size:
        return True
    else:
        return False

def solveKCG(segments, values, budget):
    n_inputs = len(values)
    ctype='I'*n_inputs
    lower_bounds = [0]*n_inputs
    upper_bounds = [1]*n_inputs
    objective = values
    colnames = ['x{:d}'.format(i) for i in  range(n_inputs)]
    weigths = []
    for s_seg in segments:
        s_weight = s_seg[1]-s_seg[0]
        weigths.append(s_weight)

    rows = []
    rows.append([colnames, weigths])
    rhs = [budget]
    sense=['L']
    for i in range(n_inputs):
        for j in range(i+1, n_inputs):
            s_seg_i = segments[i]
            s_seg_j = segments[j]
            if isOverlap(s_seg_i, s_seg_j):
                s_colnames = [colnames[i], colnames[j]]
                rows.append([s_colnames, [1, 1]])
                # rows.append([s_colnames, [1, 1]])
                rhs.append(1)
                # rhs.append(0)
                sense.append('L')
                # sense.append('G')

    n_rows = len(rows)
    row_names = ['r{:d}'.format(i) for i in range(n_rows)]
    solver = cplex.Cplex()
    solver.objective.set_sense(solver.objective.sense.maximize)
    solver.variables.add(obj=objective, lb=lower_bounds, ub=upper_bounds, types=ctype, names=colnames)
    solver.linear_constraints.add(lin_expr=rows, senses=sense, rhs=rhs, names=row_names)
    solver.solve()
    pick = solver.solution.get_values()

    picked_segments = []
    for (s_pick, s_seg)in zip(pick, segments):
        if s_pick == 1:
            picked_segments.append(s_seg)

    return picked_segments


if __name__ == "__main__":
    segments = [[10, 15], [13, 18], [6, 11]]
    values = [11, 8, 6]
    budget = 10
    solveKCG(segments, values, budget)
    print "DEBUG"

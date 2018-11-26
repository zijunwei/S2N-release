
import numpy as np

def isOverlap(seg_a, seg_b):
    left = min(seg_a[0], seg_b[0])
    right = max(seg_a[1], seg_b[1])
    union_szie = right-left
    sep_size = seg_a[1] - seg_a[0] + seg_b[1] - seg_b[0]
    if union_szie < sep_size:
        return True
    else:
        return False


def isBInA(seg_a, seg_b):
    if seg_b[0] > seg_a[0] and seg_b[1] < seg_a[1]:
        return True
    else:
        return False

def NonOverlapB(seg_a, seg_b, value_a, value_b, weight):
    new_val = value_a * weight + value_b * (1 - weight)
    if seg_b[0]<seg_a[0] and seg_b[1]>seg_b[1]:
        new_segs = [[seg_a[0], seg_b[0]], [seg_a[1], seg_b[1]]]
        new_values = [new_val, new_val]
    elif seg_b[1] > seg_a[1] and seg_b[0]<seg_a[1]:
        new_segs = [[seg_a[1], seg_b[1]]]
        new_values = [new_val]
    elif seg_b[0]<seg_a[0] and seg_b[1]>seg_a[0]:
        new_segs = [[seg_b[0], seg_a[0]]]
        new_values = [new_val]

    else:
        print "Undefined Situation, Double Check"

    return new_segs, new_values





def deoverlap_segs(segments, values, smooth_weight=0.5):
    values = np.asarray(values)
    sorted_idx = np.argsort(-values)
    values= values[sorted_idx].tolist()
    segments = [segments[i] for i in sorted_idx]


    new_segments = [segments[0]]
    new_values = [values[0]]
    values.pop(0)
    segments.pop(0)


    def inner_loop(i_segment, i_value, segments, values, smooth_weight=smooth_weight):
        FlagNonOverlap = True
        for j_idx, (j_segment, j_value) in enumerate(zip(segments, values)):
            if isOverlap(i_segment, j_segment) and not isBInA(i_segment, j_segment):
                new_segment, new_value = NonOverlapB(i_segment, j_segment, i_value, j_value, smooth_weight)
                segments.pop(j_idx)
                values.pop(j_idx)
                FlagNonOverlap = False
                return FlagNonOverlap, new_segment, new_value
        return FlagNonOverlap, None, None





    while len(segments) > 0:
        FlagNonOverlap = True
        for (i_segment, i_value) in zip(new_segments, new_values):
            # idxPop = []

            # for j_idx, (j_segment, j_value) in enumerate(zip(segments, values)):
                # if isOverlap(i_segment, j_segment) and not isBInA(i_segment, j_segment):
                    # new_segment, new_value = NonOverlapB(i_segment, j_segment, i_value, j_value, smooth_weight)
                    # new_segments.extend(new_segment)
                    # new_values.extend(new_value)
                    #
                    # idxPop.append(j_idx)
                    # FlagNonOverlap = False
                curNonOverlap, new_segment, new_value = inner_loop(i_segment, i_value, segments, values, smooth_weight)
                if not curNonOverlap:
                    new_segments.extend(new_segment)
                    new_values.extend(new_value)
                FlagNonOverlap = FlagNonOverlap and curNonOverlap
            # idxPop = idxPop[::-1]
            # for idx in idxPop:
            #     segments.pop(idx)
            #     values.pop(idx)


        if FlagNonOverlap:
            new_segments.append(segments[0])
            new_values.append(values[0])
            segments.pop(0)
            values.pop(0)


    return new_segments, new_values


if __name__ == '__main__':
    segments = [[10, 15], [13, 18], [6, 11], [20, 24]]
    values = [11, 8, 6, 15]
    smooth_weight = 0.5
    deoverlap_segs(segments, values, smooth_weight)
    print "DEBUG"


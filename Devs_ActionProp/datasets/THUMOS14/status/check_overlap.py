import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

ambiguous_class = 'Ambiguous'
user_root = os.path.expanduser('~')
datast_name = 'THUMOS14'
dataset_path = os.path.join(user_root, 'datasets',datast_name)


def getIdNames(path_root=None, isAug=False):
    if path_root is None:
        path_root = os.path.join(dataset_path, 'info')

    th14classids = []
    th14classnames = []
    target_file = open(os.path.join(path_root, 'detclasslist.txt'), 'rb')
    for line in target_file:
        elements = line.split()
        th14classids.append(int(elements[0]))
        th14classnames.append(elements[1])

    if isAug:
        th14classids.append(0)
        th14classnames.append(ambiguous_class)
    return th14classids, th14classnames


def createKeys(classids, classnames):
    id2name = {}
    name2id = {}
    for s_classid, s_classname in zip(classids, classnames):
        id2name[str(s_classid)] = s_classname
        name2id[s_classname] = s_classid

    return id2name, name2id

def IoU(inteval1, inteval2):
    i1 = [min(inteval1), max(inteval1)]
    i2 = [min(inteval2), max(inteval2)]

    b_union = [min(i1[0], i2[0]), max(i1[1], i2[1])]
    b_inter = [max(i1[0], i2[0]), min(i1[1], i2[1])]

    union = b_union[1] - b_union[0]
    intersection = b_inter[1] - b_inter[0]
    return intersection*1./union

if __name__ == '__main__':

    classids, classnames = getIdNames(isAug=True)

    annotation_directory = os.path.join(dataset_path, 'annotations')
    split = 'val'

    video_file_segments ={}
    for s_classname, class_id in zip(classnames, classids):
        target_filename = '{:s}_{:s}.txt'.format(s_classname, split)
        s_filename = os.path.join(annotation_directory, target_filename)
        s_file = open(s_filename, 'rb')
        for line in s_file:
            elements = line.strip().split()
            video_name = elements[0]
            action_idx = class_id
            start_s = float(elements[1])
            end_s = float(elements[2])
            if video_name in video_file_segments.keys():
                video_file_segments[video_name].append([start_s, end_s, action_idx])
            else:
                video_file_segments[video_name] = [[start_s, end_s, action_idx]]


    id2name, _ = createKeys(classids, classnames)
    n_instances = 0
    overlap_same_class = 0
    overlap_overall = 0
    overlap_background = 0
    # check all overlaps in single files
    for s_video_name in video_file_segments.keys():
        s_video_segments = video_file_segments[s_video_name]
        n_segments = len(s_video_segments)
        n_instances += n_segments
        for i in range(n_segments):
            i_segment = s_video_segments[i]
            for j in range(i+1, n_segments):
                j_segment = s_video_segments[j]
                ij_IOU = IoU(i_segment[0:2], j_segment[0:2])
                if ij_IOU>0:
                    overlap_overall += 1
                    print ('{:s}\t[{:.2f}, {:.2f}, {:s}]\t[{:.2f}, {:.2f}, {:s}], IOU: {:.4f}'.format(s_video_name,
                            i_segment[0], i_segment[1], id2name[str(i_segment[2])], j_segment[0], j_segment[1], id2name[str(j_segment[2])], ij_IOU))
                    if i_segment[2] == 0 or j_segment[2] == 0:
                        overlap_background += 1
                    elif i_segment[2] == j_segment[2]:
                        overlap_same_class +=1

                else:
                    continue

    print("Total number of instances: {:d}".format(n_instances))
    print("overlap of same classes: {:d}, different classes: {:d}, with Ambiguous: {:d}".format(overlap_same_class, overlap_overall, overlap_background))


    print "DG"







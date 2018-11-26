# this is a set of metrics that follows the THUMOS challenge
# http://crcv.ucf.edu/THUMOS14/download.html
import os
import sys
import numpy as np
ambiguous_class = 'Ambiguous'

thumos_val_path_root = '/mnt/data/zwei/datasets/THUMOS/TH14evalkit/groundtruth'

class DetEvent():
    def __init__(self, video_name, time_interval, class_name, class_id, confid):
        self.video_name = video_name
        self.time_interval = time_interval
        self.class_name = class_name
        self.class_id = class_id
        self.confid = confid


def getIdNames(path_root=None, isAug=False):
    if path_root is None:
        path_root = thumos_val_path_root

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


def getGTEvents(path_root=None, subset='val', isAug=True):
    if path_root is None:
        path_root = thumos_val_path_root

    th14classids, th14classnames = getIdNames(path_root=path_root, isAug=isAug)
    GTEventsList = []
    for class_id, class_name in  zip(th14classids, th14classnames):
        gtfile = os.path.join(path_root, '{:s}_{:s}.txt'.format(class_name, subset))
        for line in open(gtfile, 'rb'):
            elements = line.split()
            video_name = elements[0]
            time_inteval = [float(elements[1]), float(elements[2])]
            gtevent = DetEvent(video_name, time_inteval, class_name, class_id, confid=1)
            GTEventsList.append(gtevent)

    return GTEventsList


def getPredEvents(detfile, id2name_dict):

    predEvents = []
    for line in open(detfile, 'rb'):
        elements = line.split()
        video_name = os.path.splitext(os.path.basename(elements[0]))[0]
        start_s = float(elements[1])
        end_s = float(elements[2])
        class_id = int(elements[3])
        class_name = id2name_dict[str(class_id)]
        confid = float(elements[4])
        pred_event = DetEvent(video_name, [start_s, end_s], class_name, class_id, confid)
        predEvents.append(pred_event)

    return predEvents


def computeClassAP(gtEvents, predEvents, class_name, thres=0.1):
    gtvideonames = [x.video_name for x in gtEvents]
    detvideonames = [x.video_name for x in predEvents]
    videonames = list(set(gtvideonames+detvideonames))
    videonames.sort()


    class_gtEvents = []
    class_predEvents = []

    for s_gtEvent in gtEvents:
        if s_gtEvent.class_name == class_name:
            class_gtEvents.append(s_gtEvent)

    for s_predEvent in predEvents:
        if s_predEvent.class_name == class_name:
            class_predEvents.append(s_predEvent)

    n_pos = len(class_gtEvents)
    assert n_pos>0., '{:s} Dose Not have positive examples'.format(class_name)

    tpconf = []
    fpconf = []

    for s_videoname in videonames:
        video_gts = [gtevent for gtevent in class_gtEvents if gtevent.video_name == s_videoname]
        video_ambs = [gtevent for gtevent in class_gtEvents if gtevent.video_name == ambiguous_class]
        video_dets = [detevent for detevent in class_predEvents if detevent.video_name == s_videoname]
        if len(video_dets)>0:
            det_scores = [event.confid for event in video_dets]
            det_scores = np.asarray(det_scores)
            sorted_idxes = np.argsort(-(det_scores))
            video_dets = [video_dets[i] for i in sorted_idxes]
            det_scores = [event.confid for event in video_dets]
            det_scores = np.asarray(det_scores)
            indfree = np.ones(len(video_dets))
            indamb = np.zeros(len(video_dets))

            if len(video_gts)>0:
                gt_intevals = [event.time_interval for event in video_gts]
                det_intevals = [event.time_interval for event in video_dets]
                iou_mat = overlap_mat(gt_intevals, det_intevals)

                for k in range(len(video_gts)):
                    ind = np.nonzero(indfree)[0]
                    idx = np.argmax(iou_mat[k, ind])
                    if iou_mat[k, ind[idx]]>thres:
                        indfree[ind[idx]]=0

            if len(video_ambs):
                det_intevals = [event.time_interval for event in video_dets]
                amb_intevals = [event.time_interval for event in video_ambs]


                iou_mat = overlap_mat(amb_intevals, det_intevals)

                indamb = np.sum(iou_mat, axis=0)


            tpconf.extend(det_scores[indfree==0])
            fpconf.extend(det_scores[ np.logical_and(indfree==1,  indamb==0)])

    confs = tpconf + fpconf
    flags = np.hstack([1*np.ones(len(tpconf)), 2*np.ones(len(fpconf))])

    sort_idxes = np.argsort(-np.asarray(confs))
    tp = np.cumsum(flags[sort_idxes] == 1)
    fp = np.cumsum(flags[sort_idxes] == 2)
    rec = tp *1. / n_pos
    prec = tp * 1./ (fp + tp)
    ap = prAP(rec, prec)
    return ap


def prAP(rec, prec):
    ap = 0
    #TODO: here is problematic since it contains 6.00000000000000001 sometimes
    recallpoints = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for t in recallpoints:
        p = prec[rec>=t]
        if len(p) > 0:
            val = max(p)
        else:
            val = 0
        ap += val/len(recallpoints)
    return ap


def overlap_mat(inteval_list1, inteval_list2):
    iou_mat = np.zeros([len(inteval_list1), len(inteval_list2)])
    for i in range(len(inteval_list1)):
        for j in range(len(inteval_list2)):
            iou_mat[i, j] = IoU(inteval_list1[i], inteval_list2[j])

    return iou_mat


def IoU(inteval1, inteval2):
    i1 = [min(inteval1), max(inteval1)]
    i2 = [min(inteval2), max(inteval2)]

    b_union = [min(i1[0], i2[0]), max(i1[1], i2[1])]
    b_inter = [max(i1[0], i2[0]), min(i1[1], i2[1])]

    union = b_union[1] - b_union[0]
    intersection = b_inter[1] - b_inter[0]
    return intersection*1./union


if __name__ == '__main__':
    th14classids, th14classnames = getIdNames(isAug=False)
    id2name_dict, name2id_dict = createKeys(th14classids, th14classnames)
    gtEvents = getGTEvents()
    det_filename = '/mnt/data/zwei/datasets/THUMOS/TH14evalkit/results/Run-2-det.txt'
    predEvents = getPredEvents(det_filename, id2name_dict)
    thres_holds = [0.1, 0.2, 0.3, 0.4, 0.5]

    for s_classname in th14classnames:
        for s_thres in thres_holds:
            class_ap = computeClassAP(gtEvents, predEvents, s_classname, thres=s_thres)
            print("AP:{:.4f} at overlap: {:.2f} for {:s}".format(class_ap, s_thres, s_classname))

    print("DEBUG")




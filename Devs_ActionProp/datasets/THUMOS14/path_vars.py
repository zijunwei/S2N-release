import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

import PyUtils.load_utils as load_utils

def createKeys(classids, classnames):
    id2name = {}
    name2id = {}
    for s_classid, s_classname in zip(classids, classnames):
        id2name[str(s_classid)] = s_classname
        name2id[s_classname] = s_classid

    return id2name, name2id


class PathVars():
    def __init__(self, root_path=None):
        self.datast_name = 'THUMOS14'
        self.ambiguous_class = 'Ambiguous'
        if root_path is None:
            user_root = os.path.expanduser('~')
            self.dataset_path = os.path.join(user_root, 'datasets', self.datast_name)

        self.classlist_file = os.path.join(self.dataset_path, 'info', 'detclasslist.txt')
        self.annotation_directory = os.path.join(self.dataset_path, 'annotations')
        self.feature_directory = os.path.join(self.dataset_path, 'features')
        self.flow_directory = os.path.join(self.dataset_path, 'frameflow')
        self.video_info = self.load_video_info()

        self.classids, self.classnames = self.getIdNames()
        self.id2name, self.name2id = createKeys(self.classids, self.classnames)
        self.video_frames = load_utils.load_json(os.path.join(self.dataset_path, 'info', 'video_frames.txt'))
        self.val_annotations = self.get_annotations(split='val')
        self.tst_annotations = self.get_annotations(split='test')

    def getIdNames(self):

        th14classids = []
        th14classnames = []
        target_file = open(self.classlist_file, 'rb')
        for line in target_file:
            elements = line.split()
            th14classids.append(int(elements[0]))
            th14classnames.append(elements[1])

        th14classids.append(0)
        th14classnames.append(self.ambiguous_class)
        return th14classids, th14classnames

    def get_annotations(self, split='val'):
        video_file_segments = {}
        for s_classname, class_id in zip(self.classnames, self.classids):
            target_filename = '{:s}_{:s}.txt'.format(s_classname, split)
            s_filename = os.path.join(self.annotation_directory, target_filename)
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
        return video_file_segments

    def load_video_info(self):
        video_info_file = os.path.join(self.dataset_path, 'info', 'video_info.txt')
        video_info = load_utils.load_json(video_info_file)
        return video_info

if __name__ == '__main__':
    import scipy.io as sio
    import glob
    path_vars = PathVars()

    feature_type = 'c3d'
    feature_directory = os.path.join(path_vars.feature_directory, feature_type)
    for s_video_name in path_vars.val_annotations.keys():
        s_video_feature = sio.loadmat(os.path.join(feature_directory, '{:s}.mat'.format(s_video_name)))
        s_video_info = path_vars.video_info[s_video_name]
        n_s_video_frames = len(glob.glob(os.path.join(path_vars.flow_directory, s_video_name, '*.jpg')))/3

        n_features = len(s_video_feature['relu6'])
        s_video_framerate = s_video_info['framerate']
        s_video_duration = s_video_info['duration']
        s_feature = s_video_feature['relu6']




    print "DG"

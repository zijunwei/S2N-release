import sys
import os
# project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
# from PyUtils import load_utils
import pickle as pkl

def load_TVSumRaw():
    target_file = os.path.join(project_root, 'datasets/TVSum/ydata-tvsum50-anno.tsv')
    Annotation_Info = {}
    with open(target_file, 'rb') as f:
        for s_line in f:
            info_parts = s_line.split('\t')
            video_name = info_parts[0].strip()
            s_annotation = info_parts[2].split(',')
            digit_annotation = [int(x) for x in s_annotation]
            if video_name in Annotation_Info.keys():
                Annotation_Info[video_name].append(digit_annotation)
            else:
                Annotation_Info[video_name]=[digit_annotation]

    return Annotation_Info

if __name__ == '__main__':
    Annotation_info = load_TVSumRaw()
    save_name = os.path.join(project_root, 'datasets/TVSum/TVSumRaw.pkl')
    pkl.dump(Annotation_info, open(save_name, 'wb'))

    print("DEBUG")
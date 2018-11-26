import argparse
import os
from opsio import process_proposal_list, parse_directory,z_process_proposal_list
from opsutils import get_configs
#UPDATE: this is suitable for BNINCEPTION feature!

# parser = argparse.ArgumentParser(
#     description="Generate proposal list to be used for training")
# parser.add_argument('dataset', type=str, default='thumos14')
# parser.add_argument('frame_path', type=str, default='/home/zwei/datasets/THUMOS14/frame')

# args = parser.parse_args()
dataset = 'thumos14'
frame_path = '/home/zwei/datasets/THUMOS14/frameflow'
configs = get_configs(dataset)

norm_list_tmpl = '{}_normalized_proposal_list.txt'
out_list_tmpl = '{}_proposal_list.csv'


# if args.dataset == 'activitynet1.2':
#     key_func = lambda x: x[-11:]
# elif args.dataset == 'thumos14':
key_func = lambda x: x.split('/')[-1]
# else:
#     raise ValueError("unknown dataset {}".format(args.dataset))


# parse the folders holding the extracted frames
frame_dict = parse_directory(frame_path, key_func=key_func)

z_process_proposal_list(norm_list_tmpl.format(configs['train_list']),
                      out_list_tmpl.format(configs['train_list']), frame_dict)
z_process_proposal_list(norm_list_tmpl.format(configs['test_list']),
                      out_list_tmpl.format(configs['test_list']), frame_dict)

print("proposal lists for dataset {} are ready for training.".format(dataset))
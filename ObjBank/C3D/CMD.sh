#CUDA_VISIBLE_DEVICES=1 python extract_c3d.py \
#--model '/home/yangwang/env/models/C3D@UCF101_s1/MatFile/params.mat' \
#--saveDir 'c3d_feat/'

CUDA_VISIBLE_DEVICES=3 python extract_c3d.py \
--model '/home/yangwang/env/models/C3D@UCF101_s1/MatFile_actx/params.mat' \
--saveDir 'other/c3d_actx_feat/'

CUDA_VISIBLE_DEVICES=3 python extract_c3d.py \
--model '/home/yangwang/env/models/C3D@ActionThread/MatFile/params.mat' \
--saveDir 'other/c3d_AT_feat/'

CUDA_VISIBLE_DEVICES=3 python extract_c3d.py \
--model '/home/yangwang/env/models/C3D@ActionThread/MatFile_actx/params.mat' \
--saveDir 'other/c3d_AT_actx_feat/'

# S2N Networks

## Paper:

Z. Wei et al., “[Sequence-to-Segment Networks for Segment Detection][https://papers.nips.cc/paper/7610-sequence-to-segment-networks-for-segment-detection.pdf],” in Advances in Neural Information Processing Systems (NIPS), 2018.






## Temporal Action Proposal

### Features:

For the action proposal experiment, S2N is using reduced C3D feature similar to the [DAP][https://github.com/escorciav/daps] work:

1. we densely extract C3D fc7 feature with a window size of 16 frames to represent the center frame, (then sampling every 4 frames will be done in dataloader),
2. we then use PCA to reduce the dimension from 4096D to 500D. The reduction code is [here][https://gist.github.com/escorciav/e4125c7a8a2e52ab4c47]

The extracted features for our experiment can be downloaded from [here][https://drive.google.com/file/d/16fofE34qpYSaJRALNYKAe83qIDEhhrsi/view?usp=sharing] (4.8G)





### Evaluation and data setup:

The evaluation protocal is following [Temporal Action Detection with Structured Segment Networks][https://github.com/yjxiong/action-detection].


To process the data, you need to do the following:

1. create ground truth data using frame as unit from the normalized ground truth provided by this [work][https://github.com/yjxiong/action-detection/tree/master/data].
The conversion code in this repo is at S2N-release/Devs_ActionProp/action_det_prep/gen_proposal_list_c3dd.py


### Training:

The training code is at S2N-release/Experiments/ActionProp/ActionExp_v3_final.py, modify the paths and parameters to meet your setup
A sampling training log can be found [here][ckpts/THUMOS/ActionExp_v3_final-Release-assgin0.50-alpha0.1000-dim512-dropout0.50-seqlen90-EMD-HUG/log-2018-05-15-14-08-58.txt]

A snapshot can be found [here][https://drive.google.com/file/d/1krILLEmrUmH-6IjeO11uGhSgbc2owfl2/view?usp=sharing]

### Validation:

The validation is composed of the following steps:

1. Get the predictions from S2N-release/Experiments/ActionProp/ActionEval_classification.py
2. Compute the performance under different metrics S2N-release/Devs_ActionProp/PropEval/prop_eval_savepd2csv_ourmethods_v3_rebuttal.py (modified from
[here][https://github.com/jiyanggao/TURN-TAP])
3. Draw the figures using S2N-release/Devs_ActionProp/PropEval/prop_eval_comp2baselines_draw_loss_comparison_largefont.py

### Tuning

Currently there is not too much tuning in the parameters. If you want to get better numbers, some of the following parameters are likely important

1. threshold of determining a match between proposals and ground truths (assign)
2. loss weight between localization and classification loss (alpha)
3. Sampling overlap rate when evaluating.


## Acknowledgement

In addition to the previous code referred for evaluation, the S2N built based on the following:

1. [Pointer Networks][https://github.com/shiretzet/PointerNet/blob/master/PointerNet.py]
2. [Hungurian Loss][https://github.com/Russell91/TensorBox]



## Note:

Video Summarization code is coming out soon.

This is directly ported from my local machine so there might be issues. Please do not hestitate to raise issues if you have difficulty running it.



 
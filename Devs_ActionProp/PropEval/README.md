The TURN proposal results on THUMOS-14 are in the "pkl_files" folder. To run the proposal eval:

`python prop_eval.py TURN-C3D-16_thumos14.pkl`  or  `python prop_eval.py TURN-FLOW-16_thumos14.pkl`

The THUMOS-14 detection results for TURN+SCNN are in "thumos14_results" folder. Please use the THUMOS14 official eval codes to evaluate.


 create figures:

1. compare to baselines: prop_eval_comp2baselines_new_graph
2. generate comparison to different training prop_eval_comp2baselines_draw_loss_comparison.py

the result files are in baseline_pnt_pairs
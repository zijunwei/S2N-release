# Instructions on using vsSummDev Code

## The lastest function is PtrDep_Combine.py
### The dataloader is import vsSummDevs.Exp_vSum.vsSumLoader2 as LocalDataLoader
### The pointer network should be replaced to the network we discussed today,the sparsity will be fine in the case of video summarization. For video proposal, I think one solution is to add multiple ground truths that are slightly offset from the ground truth... But you don't have to worry about it now.
### The loss function and the matching should also be what we have discussed today

But if you think there should be some changes, we can discuss tomorrow.


## The features/videos are now uploaded to the bigmind @ /home/zwei/datasets/TVSum and SumMe respectively.  Currently we are using BNInception features, but we might need to extract dense C3D features. So perhaps the fisrt todo for you is to extract the features. The frames are saved at XX/frames where XX is either SumMe or TVSum. 
A sample feature extraction of C3D feature is @ ObjBank.C3D.extract_c3d_dense_fc7.py, also perhaps you need to reduce the dimension to 500, following THUMOS14/pca_reduction.py


Also keep in mind to split the data accordingly, perhaps according to ZhangECCV16?




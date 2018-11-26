load ../info/validation_set.mat
load ../info/test_set.mat

vid2res = struct;
for i = 1:numel(validation_videos)
    v = validation_videos(i);
    res = struct;
    res.H = v.frame_height_pixels;
    res.W = v.frame_width_pixels;
    res.T = v.video_duration_seconds;
    res.FPS = v.frame_rate_FPS;
    vid2res.(v.video_name) = res;
end

for i = 1:numel(test_videos)
    v = test_videos(i);
    res = struct;
    res.H = v.frame_height_pixels;
    res.W = v.frame_width_pixels;
    res.T = v.video_duration_seconds;
    res.FPS = v.frame_rate_FPS;
    vid2res.(v.video_name) = res;
end

save('../info/vid2res.mat', 'vid2res');

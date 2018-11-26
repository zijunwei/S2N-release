% get video list and video resolution
dataRoot = '/nfs/bigeye/yangwang/DataSets/THUMOS14/';
load ../info/anno_instances.mat
load ../info/vid2res.mat

% iterate over videos
allVids = unique([val.videos; tst.videos]);
for i = 1:numel(allVids)
    id = allVids{i};
    if id(7)=='v'
        format = '.mpeg'; % for valiation
    else
        format = '.mp4';  % for test
    end
    video = [id, format];

    % where to store the frames
    frameDir = sprintf('../frame/%s/', id);
    if exist(frameDir, 'dir') continue; end
    system(['mkdir -p ', frameDir]);

    % prepare clip info
    clip.file = [dataRoot,'videos/',video];
    clip.flip = false;

    % set opts to extract frames @ claimed resolution: most are [ 180p x 320p x 25~30fps ]
    % ffmpegBin, fps, newH, newW, frmPrefix, frmExt
    res = vid2res.(id);
    opts = [];
    opts.ffmpegBin = '/home/minhhoai/local/bin/ffmpeg';
    opts.fps = res.FPS;
    opts.newH = res.H;
    opts.newW = res.W;
    opts.frmPrefix = 'i_';
    opts.frmExt = 'jpg';

    ML_VidClip.extFrms(clip, frameDir, opts);
    system(['rm -f ',frameDir,'/*.log']);
end

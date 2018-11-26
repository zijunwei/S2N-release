% TOTAL=412; COUNT=8; STEP=52;

% get video list and video resolution
dataRoot = '/nfs/bigeye/yangwang/DataSets/THUMOS14/';
load ../info/anno_instances.mat
load ../info/vid2res.mat

% iterate over videos
allVids = unique([val.videos; tst.videos]);
for i = BatchInx(TOTAL, COUNT, STEP)
    id = allVids{i};
    if id(7)=='v'
        format = '.mpeg'; % for valiation
    else
        format = '.mp4';  % for test
    end
    video = [id, format];

    % prepare clip info
    clip.file = [dataRoot,'videos/',video];
    clip.flip = false;

    % reformat video to claimed resolution: most are [ 180p x 320p x 25~30fps ]
    % ffmpegBin, fps, newH, newW, frmPrefix, frmExt
    res = vid2res.(id);
    opts = [];
    opts.ffmpegBin = '/home/minhhoai/local/bin/ffmpeg';
    opts.fps = res.FPS;
    opts.newH = res.H;
    opts.newW = res.W;
    
    system('mkdir -p ../newvideos/')    % mkdir first
    newVidFile = sprintf('../newvideos/%s',video);
    system(sprintf('rm -f %s', newVidFile));    % rm old file, extVid() will stall upon overwritting
    ML_VidClip.extVid(clip, newVidFile, opts);

    % extract frames & flows
    frameDir = sprintf('../frameflow/%s/', id);
    if exist(frameDir, 'dir') continue; end
    system(['mkdir -p ', frameDir]);

    denseFlow = '/home/yangwang/env/c/dependences/denseFlow_gpu/denseFlow_gpu';
    file1 = newVidFile;
    file2 = [frameDir,'x'];
    file3 = [frameDir,'y'];
    file4 = [frameDir,'i'];
    cmd = sprintf('%s -f=%s -x=%s -y=%s -i=%s -b=20 -t=1',denseFlow,file1,file2,file3,file4);
    system(cmd);

    % sanity-check, in case of failure
    nFrm=length(dir([frameDir,'/*.jpg']));
    if (nFrm == 0)
        system(sprintf('echo "%s" >> zeroframe.error',file1));
    end
end

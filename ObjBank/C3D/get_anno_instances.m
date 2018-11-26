actions = textread('../info/annotated_classes.txt','%s');

val.videos = [];
val.starts = [];
val.ends = [];
val.labels = [];
for i = 1:numel(actions)
    action = actions{i};
    [~, actionID] = ismember(action, actions);
    [videos_, starts_, ends_] = textread(sprintf('../annos/%s_val.txt',action),'%s%f%f');
    labels_ = actionID * ones(size(videos_));
    val.videos = [val.videos; videos_];
    val.starts = [val.starts; starts_];
    val.ends   = [val.ends; ends_];
    val.labels = [val.labels; labels_];
end

tst.videos = [];
tst.starts = [];
tst.ends = [];
tst.labels = [];
for i = 1:numel(actions)
    action = actions{i};
    [~, actionID] = ismember(action, actions);
    [videos_, starts_, ends_] = textread(sprintf('../annos/%s_test.txt',action),'%s%f%f');
    labels_ = actionID * ones(size(videos_));
    tst.videos = [tst.videos; videos_];
    tst.starts = [tst.starts; starts_];
    tst.ends   = [tst.ends; ends_];
    tst.labels = [tst.labels; labels_];
end

save('../info/anno_instances.mat','actions','val','tst');

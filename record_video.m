%RECORD_VIDEO  Record an MPEG-4 of one proposed (transfer) simulation.
%   Runs MAIN_LEARNER once with cfg.record = 1 and writes the reaching movements
%   to result/ (.mp4 on Windows/macOS, .avi on Linux, see VIDEO_PROFILE).
%
%   The recorded run is the first simulation of the batch (simN = 1, hence the
%   same seed REPEAT_MAIN uses for it), with the hand drawn at random every
%   session and the target fixed: (7,5) at baseline and shifted by cfg.dis in
%   cfg.dir while the prism is on (randhand == 1).
%
%   Run this from the MATLAB desktop.
%
%   See also DEFAULT_CONFIG, MAIN_LEARNER, REPEAT_MAIN.

clear;

cfg = default_config( ...
    'sim_type', "transfer", ...  % "proposed" learner
    'randhand', 1,          ...  % random hand, fixed target
    'simN',     1,          ...  % first simulation of the batch
    'record',   1,          ...  % implies cfg.draw = 1
    'saveMat',  0);              % video only; set to 1 to keep the snapshots too

fprintf('recording seed %d ...\n', cfg.seed);
rslt = main_learner(cfg);
fprintf('done: %s\n', rslt.video_File);

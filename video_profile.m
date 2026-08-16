function [profile, ext] = video_profile()
%VIDEO_PROFILE  VideoWriter profile usable on the current platform.
%   [PROFILE,EXT] = VIDEO_PROFILE() returns the profile name to hand to
%   VideoWriter and the extension VideoWriter appends to the file name.
%
%   The 'MPEG-4' profile ships only with Windows and macOS; on Linux
%   VideoWriter errors with "the specified profile is not valid", so the
%   recording falls back to 'Motion JPEG AVI' (larger files, same frames).
%
%   See also MAIN_LEARNER, RECORD_VIDEO.

if ispc || ismac
    profile = 'MPEG-4';
    ext     = ".mp4";
else
    profile = 'Motion JPEG AVI';
    ext     = ".avi";
end
end

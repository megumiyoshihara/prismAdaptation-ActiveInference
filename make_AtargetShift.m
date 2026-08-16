function A = make_AtargetShift(size, direction, distance)
%MAKE_ATARGETSHIFT  Likelihood matrix that shifts the perceived target.
%   A = MAKE_ATARGETSHIFT(SIZE,DIRECTION,DISTANCE) builds the Nm^2 x Nm^2 prism
%   shift over a SIZE x SIZE monitor, for DIRECTION 'up', 'down', 'left' or
%   'right'.  Positions pushed past the edge pile up in the leading row of the
%   band, so the shifted target never leaves the monitor.
%
%   See also MAIN_LEARNER, MAKE_INITPOS_MD.
I = eye(size, 'single');
M = zeros(size, 'single');
M(1, 1:distance) = ones(1, distance, 'single');
M(1:size - distance, distance+1:size) = eye(size - distance, 'single');
if strcmp(direction, 'up')
    A = kron(I, M);
elseif strcmp(direction, 'down')
    A = kron(I, flip(flip(M,1),2));
elseif strcmp(direction, 'left')
    A = kron(M, I);
elseif strcmp(direction, 'right')
    A = kron(flip(flip(M,1),2),I);
end
end
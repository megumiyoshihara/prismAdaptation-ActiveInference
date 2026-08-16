function posSet = make_initPos_MD(Nsim,Nsession,MD,shiftDir)
%MAKE_INITPOS_MD  Per-simulation target pair and hand trajectory.
%   POSSET = MAKE_INITPOS_MD(NSIM,NSESSION,MD) draws, for every simulation, an
%   exposure target away from the border and a baseline target at Manhattan
%   distance MD from it, in a random direction.
%
%   POSSET = MAKE_INITPOS_MD(...,SHIFTDIR) fixes that direction instead.
%   SHIFTDIR is "random" (the default), "right", "left", "up" or "down" and
%   uses the same sense as MAKE_ATARGETSHIFT
%   Since this uses the same directional definitions as `MAKE_ATARGETSHIFT`
%   , passing `cfg.dir` here allows you to generate positions consistent with
%   the shift matrix constructed by `MAIN_LEARNER` when `randhand == 3`.
%
%   See also PREPARE_POSITIONS, MAKE_ATARGETSHIFT, DEFAULT_CONFIG.
arguments
    Nsim
    Nsession
    MD
    shiftDir (1,1) string = "random"
end
seed = 63217438;
rng(seed);

Nm = 10; % size of monitor

% Targets are drawn from the interior only, so that a shift of MD in any
% direction can still land off the border.
XList = 2:9;
YList = 2:9;

% A fixed direction constrains where the exposure target may sit, so that the
% shifted (baseline) target still lands off the border.  "random" leaves both
% lists at the full interior range.
switch shiftDir
    case "random"  % nothing to restrict
    case "right", XList = 2:(Nm-1-MD);
    case "left",  XList = (2+MD):(Nm-1);
    case "down",  YList = 2:(Nm-1-MD);
    case "up",    YList = (2+MD):(Nm-1);
    otherwise
        error("make_initPos_MD:badShiftDir", ...
              'shiftDir must be "random", "right", "left", "up" or "down"');
end
if isempty(XList) || isempty(YList)
    error("make_initPos_MD:noRoom", ...
          'MD = %d leaves no interior target pair for a fixed "%s" shift on a %dx%d grid', ...
          MD, shiftDir, Nm, Nm);
end

function T = manhattanInteriorWithDeltas(x0,y0,n,gridSize)
    % Return interior grid points at Manhattan distance n from (x0,y0),
    % excluding border (1 or gridSize). Also return integer x/y separations.
    % Inputs:
    %   x0,y0    - target coordinates (1-based)
    %   n        - Manhattan distance (nonnegative integer)
    %   gridSize - optional, default 10
    % Output:
    %   T - table with columns:
    %       x   : x coordinate
    %       y   : y coordinate
    %       dx  : x - x0 (integer, signed)
    %       dy  : y - y0 (integer, signed)
    %       dist: Manhattan distance (should equal n)
    
    if nargin < 4
        gridSize = 10;
    end
    
    % create grid indices (x: columns, y: rows)
    [xGrid,yGrid] = meshgrid(1:gridSize, 1:gridSize);
    
    % Manhattan distance
    dist = abs(xGrid - x0) + abs(yGrid - y0);
    
    % mask for distance == n and not on border
    mask = (dist == n) & ...
           ~(xGrid == 1 | xGrid == gridSize | yGrid == 1 | yGrid == gridSize);
    
    % extract coordinates
    xs = xGrid(mask);
    ys = yGrid(mask);
    
    % compute deltas (signed integers)
    dx = xs - x0;
    dy = ys - y0;
    distCols = dist(mask);
    
    % assemble as table for readability
    T = table(xs, ys, dx, dy, distCols, ...
        'VariableNames', {'x','y','dx','dy','dist'});
end


% Per simulation: draw an exposure target away from the border, pick a baseline
% target at Manhattan distance MD from it (again off the border), then build the
% shift matrix A by composing the horizontal and the vertical shift.

for i = 1:Nsim
    idx = randperm(length(XList), 1);
    X = XList(idx(1));
    idy = randperm(length(YList), 1);
    Y = YList(idy);

    T = manhattanInteriorWithDeltas(X,Y,MD,Nm);
    % Keep only the candidate lying in the requested direction; the signs match
    % the make_AtargetShift calls below.
    switch shiftDir
        case "right", T = T(T.dx > 0 & T.dy == 0, :);
        case "left",  T = T(T.dx < 0 & T.dy == 0, :);
        case "down",  T = T(T.dy > 0 & T.dx == 0, :);
        case "up",    T = T(T.dy < 0 & T.dx == 0, :);
    end
    assert(~isempty(T), "make_initPos_MD:noCandidate", ...
           'no interior point at distance %d towards %s from (%d,%d)', ...
           MD, shiftDir, X, Y);

    idt = randperm(size(T,1),1);
    sX = T.x(idt); % Shifted X coordinate based on selected index
    sY = T.y(idt); % Shifted Y coordinate based on selected index

    posSet(i).targetX_Exposure = X;
    posSet(i).targetY_Exposure = Y;

    posSet(i).shiftedX_Exposure = sX;
    posSet(i).targetX_Baseline = sX;
    

    posSet(i).shiftedY_Exposure = sY;
    posSet(i).targetY_Baseline = sY;

    posSet(i).dis = MD;
    posSet(i).shiftDir = shiftDir;
    A = eye(Nm^2);
    if T.dx(idt) < 0
        A = A*make_AtargetShift(Nm,'left',abs(T.dx(idt)));
    elseif T.dx(idt) > 0
        A = A*make_AtargetShift(Nm,'right',abs(T.dx(idt)));
    end
    if T.dy(idt) < 0
        A = A*make_AtargetShift(Nm,'up',abs(T.dy(idt)));
    elseif T.dy(idt) > 0
        A = A*make_AtargetShift(Nm,'down',abs(T.dy(idt)));
    end
    posSet(i).A = A;


    for j=1:Nsession
        if j == 1
            targetX = sX;
            targetY = sY;
        elseif j == 141
            targetX = X;
            targetY = Y;
    
        elseif j == 171
            targetX =sX;
            targetY = sY;
        end
    
        xlist = cat(2, 1:targetX-1, targetX+1:Nm);
        ylist = cat(2, 1:targetY-1, targetY+1:Nm);
        Xvec = zeros(Nm-1,1);
        Yvec = zeros(Nm-1,1);
        Xvec(randi(Nm-1, 1),1)=1;
        Yvec(randi(Nm-1, 1),1)=1;
    
        handX = xlist *Xvec;
        handY = ylist *Yvec;
    
        posSet(i).Hand(j)=(handX-1)*Nm^3+(handY-1)*Nm^2+(targetX-1)*Nm+targetY;
    end


end

end
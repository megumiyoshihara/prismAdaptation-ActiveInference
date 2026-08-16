function prepare_positions(cfg, force)
%PREPARE_POSITIONS  Generate the initial position files and their shortest distances.
%   PREPARE_POSITIONS(CFG) writes, unless they already exist:
%     cfg.initHand_File        initial hand positions for a fixed target
%     cfg.InitPos_File         randomised target/shift positions at distance cfg.MD
%     cfg.shortestDisHand_File shortest distance for the fixed-target hand positions
%     cfg.shortestDis_File     shortest hand-to-target distance per session
%
%   PREPARE_POSITIONS(CFG, TRUE) regenerates them even if they exist.
%
%   See also PREPARE_ALL, PREPARE_PRETRAINING, MAKE_INITHAND, MAKE_INITPOS_MD.

arguments
    cfg   (1,1) struct
    force (1,1) logical = false
end

Nm = cfg.Nm;

%% Initial hand positions with a fixed target ----------------------------
if force || ~isfile(cfg.initHand_File)
    fprintf('  generating %s ...\n', cfg.initHand_File);
    Hand = make_initHand(cfg.Nsim, cfg.Nsession);
    save(cfg.initHand_File, "Hand", '-v7.3');
else
    fprintf('  skipped %s (already exists)\n', cfg.initHand_File);
    loaded = load(cfg.initHand_File, "Hand");
    Hand   = loaded.Hand;
end

%% Randomised target positions at Manhattan distance cfg.MD --------------
if force || ~isfile(cfg.InitPos_File)
    fprintf('  generating %s ...\n', cfg.InitPos_File);
    if cfg.randhand == 3
        % randhand 3 shifts every simulation with the same matrix,
        % make_AtargetShift(Nm, cfg.dir, cfg.dis) in MAIN_LEARNER, so the target
        % pair has to be shifted the same way -- cfg.dir at distance cfg.dis.
        posSet = make_initPos_MD(cfg.Nsim, cfg.Nsession, cfg.dis, cfg.dir);
    else
        posSet = make_initPos_MD(cfg.Nsim, cfg.Nsession, cfg.MD, cfg.shiftDir);
    end
    save(cfg.InitPos_File, "posSet", '-v7.3');
else
    fprintf('  skipped %s (already exists)\n', cfg.InitPos_File);
    loaded = load(cfg.InitPos_File, "posSet");
    posSet = loaded.posSet;
end

%% Shortest hand-to-target distance per session --------------------------
if force || ~isfile(cfg.shortestDisHand_File)
    fprintf('  generating %s ...\n', cfg.shortestDisHand_File);
    shortestDis_hand = shortest_distance(Hand.s', Nm);
    save(cfg.shortestDisHand_File, "shortestDis_hand", '-v7.3');
else
    fprintf('  skipped %s (already exists)\n', cfg.shortestDisHand_File);
end

if force || ~isfile(cfg.shortestDis_File)
    fprintf('  generating %s ...\n', cfg.shortestDis_File);
    shortestDis_pos = zeros(cfg.Nsession, cfg.Nsim);
    for i = 1:cfg.Nsim
        shortestDis_pos(:,i) = shortest_distance(posSet(i).Hand(:), Nm);
    end
    save(cfg.shortestDis_File, "shortestDis_pos", '-v7.3');
else
    fprintf('  skipped %s (already exists)\n', cfg.shortestDis_File);
end
end


function d = shortest_distance(s, Nm)
%SHORTEST_DISTANCE  Manhattan steps from hand to target for a packed state index.
%   The state index packs both positions as
%       s = (handX-1)*Nm^3 + (handY-1)*Nm^2 + (targetX-1)*Nm + targetY
%   so the target part is mod(s, Nm^2) and the hand part is fix(s / Nm^2).
%   The -1 corrects for targetY being 1-based while handY-1 is 0-based.

targetPart = mod(s, Nm^2);
handPart   = fix(s / Nm^2);
d = abs(mod(targetPart, Nm) - mod(handPart, Nm) - 1) + ...
    abs(fix(targetPart / Nm) - fix(handPart / Nm));
end

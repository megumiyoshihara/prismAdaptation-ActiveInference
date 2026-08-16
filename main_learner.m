function rslt = main_learner(cfg)
%MAIN_LEARNER  Run one prism-adaptation simulation for a naive or transfer learner.
%   RSLT = MAIN_LEARNER(CFG) simulates CFG.Nsession sessions of a reaching task
%   under an active-inference agent.  The prism is put on at session
%   CFG.exposureStart and taken off at CFG.removalStart.  CFG comes from
%   DEFAULT_CONFIG; only CFG.simN and CFG.seed normally change between calls.
%
%   The naive and the transfer learner share this whole routine.  They differ
%   only in how the goal prior C is initialised, evaluated and updated, and
%   those three points are delegated to LEARNER_INIT, LEARNER_POLICY and
%   LEARNER_UPDATE.
%
%   RSLT holds the per-session trajectories (info_s, info_o, tList) and, for the
%   transfer learner, the mixing weights lambda.  With CFG.saveMat the full
%   session snapshots are written to result/ as well.
%
%   See also DEFAULT_CONFIG, REPEAT_MAIN, PREPARE_ALL.

%% Setup -----------------------------------------------------------------
learnedC = [];
if cfg.sim_type == "transfer"
    % Pretrained goal priors: one C matrix per pretrained target position,
    % blurred by cfg.blur.
    learnedC = load_pretrained_policy(cfg);
end

% Used in the file name when the seed is not fixed.
id = randi(65536);
if cfg.useseed == 1
    rng(cfg.seed);
end

Nm    = cfg.Nm;
Ns    = cfg.Ns;
No    = cfg.No;
Nd    = cfg.Nd;
handX = cfg.handX;
handY = cfg.handY;

rslt = struct();

%% Target and hand positions ---------------------------------------------
switch cfg.randhand
    case {3, 4}
        % Baseline and exposure targets are replayed from a prepared file.
        loaded = load(cfg.InitPos_File, "posSet");
        posSet = loaded.posSet;
        rslt.pos_File     = cfg.InitPos_File;
        targetX_Baseline  = posSet(cfg.simN).targetX_Baseline;
        targetY_Baseline  = posSet(cfg.simN).targetY_Baseline;
        targetX_Exposure  = posSet(cfg.simN).targetX_Exposure;
        targetY_Exposure  = posSet(cfg.simN).targetY_Exposure;
    otherwise
        % Fixed target; the exposure target is the baseline one shifted by dis.
        if cfg.randhand == 2
            loaded = load(cfg.InitPos_File, "Hand");
            Hand   = loaded.Hand;
            rslt.Hand_File = cfg.InitPos_File;
        end
        targetX_Baseline = 7;
        targetY_Baseline = 5;
        switch cfg.dir
            case "down"
                targetX_Exposure = targetX_Baseline;
                targetY_Exposure = max(1, targetY_Baseline - cfg.dis);
            case "up"
                targetX_Exposure = targetX_Baseline;
                targetY_Exposure = min(Nm, targetY_Baseline + cfg.dis);
            case "right"
                targetX_Exposure = max(1, targetX_Baseline - cfg.dis);
                targetY_Exposure = targetY_Baseline;
            case "left"
                targetX_Exposure = min(Nm, targetX_Baseline + cfg.dis);
                targetY_Exposure = targetY_Baseline;
            otherwise
                error("main_learner:unknownDirection", ...
                      "dir must be up/down/left/right, got ""%s"".", cfg.dir);
        end
        firstS = (handX-1)*Nm^3 + (handY-1)*Nm^2 + (targetX_Baseline-1)*Nm + targetY_Baseline;
end

%% Output file name ------------------------------------------------------
Name_folder  = "result/";
Name_rndHand = ["THfixed", "TfixedHrand", "TfixedHgiven", "THgiven", "MDrandomTHgiven" + cfg.MD];
Name_NumTS   = "T" + cfg.T + "S" + cfg.Nsession;
Name_Seed    = ["seedRid" + id, "seed" + cfg.seed];

% What this run was: positions, baseline target, schedule and seed.  This is
% what tells two runs apart.
condition = Name_rndHand(cfg.randhand+1) + "_target" + targetX_Baseline + targetY_Baseline + ...
            "_" + Name_NumTS + "_" + Name_Seed(cfg.useseed+1);

% With cfg.saveMat the run leaves session snapshots; without it the only output
% is the video RECORD_VIDEO asks for.
if cfg.saveMat == 1
    role = "snapshot";
else
    role = "video";
end
filename = Name_folder + role + "_main_" + cfg.sim_type + "_" + condition;
if ~exist(Name_folder, 'dir')
    mkdir(Name_folder);
end

%% Generative model ------------------------------------------------------
% Without the prism A is the identity; with the prism A shifts the target.
Ahand   = eye(Nm^2, Nm^2, 'single');
Atarget = make_AtargetShift(Nm, cfg.dir, cfg.dis);
if cfg.randhand == 4
    % Each simulation has its own target shift, prepared by MAKE_INITPOS_MD.
    Atarget = posSet(cfg.simN).A;
end
Ashift = kron(Ahand, Atarget);

B_target = eye(Nm^2, Nm^2);
B_hand   = make_Bhand(Nm);
B        = kron(B_hand, B_target);
D        = ones(Ns, 1, 'single') * 0.5 / Ns;
E        = kron(repmat(cfg.prior, cfg.Nu, 1), ones(Nd/cfg.Nu, 1) / (Nd/cfg.Nu));

%% Posterior initialisation ----------------------------------------------
Q(1) = learner_init(cfg, B, D, E, learnedC);
if cfg.sim_type == "transfer"
    rslt.pretrained_policy_File = cfg.pretrained_policy_File;
end
rslt.seed = cfg.seed;

%% Session loop -----------------------------------------------------------
distance = zeros(1, cfg.T);

if cfg.record == 1
    [profile, ext] = video_profile();
    v = VideoWriter(filename, profile);
    open(v);
    rslt.video_File = filename + ext;
end
if cfg.draw == 1
    figure();
end

for h = 1:cfg.Nsession
    if h == 1
        shift   = 0;
        targetX = targetX_Baseline;
        targetY = targetY_Baseline;
    elseif h == cfg.exposureStart
        % The prism is on from exposureStart until removalStart.
        shift   = 1;
        targetX = targetX_Exposure;
        targetY = targetY_Exposure;
        firstS  = (handX-1)*Nm^3 + (handY-1)*Nm^2 + (targetX-1)*Nm + targetY;
    elseif h == cfg.removalStart
        % The prism is removed for the remaining sessions.
        shift   = 0;
        targetX = targetX_Baseline;
        targetY = targetY_Baseline;
        firstS  = (handX-1)*Nm^3 + (handY-1)*Nm^2 + (targetX-1)*Nm + targetY;
    end

    if shift == 0
        A = eye(No, Ns, 'single');
    else
        A = Ashift;
    end

    [~, qlnA]  = param_normalization(Q(h).qa, "A");
    [~, qlnB]  = param_normalization(Q(h).qb, "A");
    [qc, qlnC] = learner_policy(cfg, Q(h));
    lnE = log(max(10^-6, Q(h).E));

    % Hand and target position at the first step of the session.
    switch cfg.randhand
        case 0   % fixed at (handX, handY)
            Q(h).s(firstS, 1) = 1;
        case 1   % drawn at random
            Q(h).s(:, 1) = make_randHandS(Nm, targetX, targetY);
        case 2   % replayed from the prepared "Hand" array
            Q(h).s(:, 1) = zeros(Ns, 1);
            Q(h).s(Hand.s(cfg.simN, h), 1) = 1;
        case {3, 4}  % replayed from the prepared posSet
            Q(h).s(:, 1) = zeros(Ns, 1);
            Q(h).s(posSet(cfg.simN).Hand(h)) = 1;
    end

    Q(h).o(:,1) = mnrnd(1, A * Q(h).s(:,1));
    distance(1) = manhattan_distance(Nm, Q(h).s(:,1));
    Q(h).u(1,1) = 1;
    Q(h).d(1,1) = 1;
    if cfg.draw == 1
        C4disp = Cfordisplay(Nm, qc);
    end

    rslt.info_s(h,1) = find(Q(h).s(:,1) == 1);
    rslt.info_o(h,1) = find(Q(h).o(:,1) == 1);

    for t = 2:cfg.T
        % generative process
        Q(h).s(:,t) = mnrnd(1, B * calc_DxS(Q(h).u(:,t-1), Q(h).s(:,t-1)));
        Q(h).o(:,t) = mnrnd(1, A * Q(h).s(:,t));

        % inference
        Q(h).vs(:,t) = qlnA' * Q(h).o(:,t) + qlnB * calc_DxS(Q(h).u(:,t-1), Q(h).qs(:,t-1));
        Q(h).vd(:,t) = qlnC * Q(h).qs(:,t-1) + lnE;
        Q(h).qs(:,t) = exp(Q(h).vs(:,t)-max(Q(h).vs(:,t))) ./ sum(exp(Q(h).vs(:,t)-max(Q(h).vs(:,t))));
        Q(h).qd(:,t) = exp(Q(h).vd(:,t)-max(Q(h).vd(:,t))) ./ sum(exp(Q(h).vd(:,t)-max(Q(h).vd(:,t))));

        % decision
        Q(h).d(:,t) = mnrnd(1, Q(h).qd(:,t));
        Q(h).u(:,t) = kron(eye(cfg.Nu), ones(1, Nd/cfg.Nu)) * Q(h).d(:,t);

        % risk: 0 when the hand got closer to the target, 1 when it moved away
        nowMd = manhattan_distance(Nm, Q(h).s(:,t));
        preMd = distance(t-1);
        distance(t) = nowMd;
        if preMd > nowMd
            Q(h).G(t) = 0;
        elseif preMd < nowMd
            Q(h).G(t) = 1;
        else
            Q(h).G(t) = 0.5;
        end

        if cfg.draw == 1
            figure_output_main(Nm, Q(h).s(:,t), targetX, targetY, C4disp);
            title(['session', num2str(h), ', trial', num2str(t)]);
            drawnow;
        end
        if cfg.record == 1
            writeVideo(v, getframe(gcf));
        end

        rslt.info_s(h,t) = find(Q(h).s(:,t) == 1);
        rslt.info_o(h,t) = find(Q(h).o(:,t) == 1);

        % the session ends as soon as the hand reaches the target
        if nowMd == 0
            break;
        end
    end

    Q(h).t = t;

    % Risk over the partition of Manhattan-distance changes.  The half-width is
    % cfg.risk_range divided by cfg.risk_divisor; see DEFAULT_CONFIG.
    risk_high  = 0.5 + cfg.risk_halfwidth;
    risk_low   = 0.5 - cfg.risk_halfwidth;
    risk       = [risk_high, risk_high, risk_high, risk_high, risk_high, risk_low, risk_low];
    partition  = [-1, 0, 1];
    Q(h).Gamma = calc_risk(Q(h), cfg.range, partition, risk);

    % learning
    [Qnext, transientFields] = learner_update(cfg, Q(h));

    if cfg.sim_type == "transfer"
        rslt.lambda(h,:) = Q(h).lambda;
    end
    rslt.tList(h) = Q(h).t;
    switch cfg.sim_type
        case "naive",    rslt.qcList(h).qc = Q(h).qc;
        case "transfer", rslt.qcList(h).qc = Q(h).qc1;
    end
    rslt.risk = risk;

    % Keep a full snapshot every tenth session and a trimmed one otherwise,
    % then drop Q so only the next session's posterior stays in memory.
    if h == 1 || rem(h, 10) == 0
        Qkeep1(h) = Q(h); %#ok<AGROW>
    end
    Qkeep2(h) = rmfield(Q(h), transientFields); %#ok<AGROW>
    clear Q;
    Q(h+1) = Qnext;
end

if cfg.record == 1
    close(v);
end

if cfg.saveMat == 1
    save(filename + ".mat", 'rslt', 'Qkeep1', 'Qkeep2', 'cfg', 'filename', '-v7.3');
end
end

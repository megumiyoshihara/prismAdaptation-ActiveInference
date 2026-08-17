function rslt = validation_learner(cfg)
%VALIDATION_LEARNER  Run one 3x3 validation simulation of C learning.
%   RSLT = VALIDATION_LEARNER(CFG) simulates CFG.Nsession sessions of reaching
%   towards a fixed target on an Nm x Nm field, with no prism.  CFG comes from
%   VALIDATION_CONFIG; normally only CFG.simN and CFG.seed change between calls.
%
%   The point of this experiment is to watch how the goal prior C is learned,
%   session by session, under a random walk.  Two learners are trained side by
%   side on the SAME trajectory so their learning curves are comparable:
%
%     transfer  learns lambda, the mixing weights over the pretrained C stack;
%               its C is recovered as exp(sum_k lambda_k * log preqC_k).
%     naive     learns qcValid, its C, directly.
%
%   Only those two quantities survive from one session to the next -- Q_state is
%   rebuilt every session, so qa and qb are never carried over.  The acting
%   policy qlnC1 is computed once from the uniform qc1Init and is never updated,
%   so behaviour stays a random walk: what is learned deliberately does not feed
%   back into what the agent does.  For the same reason a session always runs
%   all CFG.T trials and never stops early when the target is reached, which
%   keeps the amount of experience identical across sessions.
%
%   See also VALIDATION_CONFIG, REPEAT_VALIDATION, PREPARE_VALIDATION.

arguments
    cfg (1,1) struct
end

%% Setup -----------------------------------------------------------------
learnedC = load_pretrained_policy(cfg);
Nlambda  = size(learnedC, 3);

Nm = cfg.Nm;
Ns = cfg.Ns;
No = cfg.No;
Nd = cfg.Nd;
T  = cfg.T;
targetX = cfg.targetX;
targetY = cfg.targetY;

% Used in the file name when the seed is not fixed.
id = randi(65536);
if cfg.useseed == 1
    rng(cfg.seed);
end

rslt = struct();
rslt.pretrained_policy_File = cfg.pretrained_policy_File;
rslt.seed = cfg.seed;

% Start of the hand when cfg.randhand == 0. The state indexes the hand only.
firstS = (cfg.handX-1)*Nm + cfg.handY;

%% Output file name ------------------------------------------------------
Name_folder  = "result/";
Name_rndHand = ["Hfixed", "Hrandom"];
Name_NumTS   = "T" + T + "S" + cfg.Nsession;
Name_Seed    = ["seedRid" + id, "seed" + cfg.seed];

condition = Name_rndHand(cfg.randhand+1) + "_Ttgt" + targetX + targetY + ...
            "_" + Name_NumTS + "_" + Name_Seed(cfg.useseed+1);

filename = Name_folder + "snapshot_valid_" + condition;
if ~exist(Name_folder, 'dir')
    mkdir(Name_folder);
end

%% Generative model ------------------------------------------------------
% No prism, so the likelihood is the identity for every session.
A = eye(No, Ns, 'single');
B = make_Bhand(Nm);
D = ones(Ns, 1, 'single') * 0.5 / Ns;
E = kron(repmat(cfg.prior, cfg.Nu, 1), ones(Nd/cfg.Nu, 1) / (Nd/cfg.Nu));

% Log of each pretrained policy, the basis the transfer learner mixes over.
preqlnC = zeros(Nd, Ns, Nlambda);
for i = 1:Nlambda
    preqC = param_normalization(learnedC(:,:,i), "A");
    preqlnC(:,:,i) = log(preqC);
end

%% Posterior initialisation ----------------------------------------------
qaInit  = ones(No, Ns, 'single') * 100^(-1);
qaInit  = qaInit + eye(No, Ns);
qbInit  = ones(size(B), 'single') * 100^(0);
qc1Init = ones(Nd, Ns, 'single') * 10^(0);

% The two learned quantities, and the only state carried across sessions.
lambda   = ones(1, Nlambda, 'single') / Nlambda;
qcValid0 = ones(Nd, Ns, 'single') * 10^(0);
qcValid  = qcValid0;

Q_state = transfer_init(qaInit, qbInit, qc1Init, learnedC, lambda, D, E, T);

% The acting policy. Computed once: it is intentionally never updated, so the
% agent keeps exploring uniformly while C is being learned.
[~, qlnA]  = param_normalization(Q_state.qa, "A");
[~, qlnB]  = param_normalization(Q_state.qb, "A");
[qc1, qlnC1] = param_normalization(Q_state.qc1, "A");
lnE = log(max(10^-6, Q_state.E));

rslt.qcValid0 = qcValid0;
rslt.lambda0  = lambda;
rslt.qa0      = qaInit;

lambdaList  = zeros(cfg.Nsession, Nlambda);
vlambdaList = zeros(cfg.Nsession, Nlambda);
distance    = zeros(1, T);

if cfg.draw == 1
    figure();
    % Loop invariant: qc1 never changes, so the policy map is drawn once.
    C4disp = Cfordisplay_validation(Nm, qc1);
end

%% Session loop -----------------------------------------------------------
for h = 1:cfg.Nsession
    % Rebuilt every session: only lambda and qcValid persist.
    Q_state = transfer_init(qaInit, qbInit, qc1Init, learnedC, lambda, D, E, T);

    if cfg.randhand == 0
        Q_state.s(firstS, 1) = 1;
    else
        Q_state.s(:, 1) = make_randHandS_validation(Nm, targetX, targetY);
    end

    Q_state.o(:,1) = mnrnd(1, A * Q_state.s(:,1));
    distance(1)    = manhattan_distance_validation(Nm, Q_state.s(:,1), targetX, targetY);
    Q_state.u(1,1) = 1;
    Q_state.d(1,1) = 1;

    if cfg.record_traces == 1
        rslt.info_s(h,1) = find(Q_state.s(:,1) == 1);
        rslt.info_o(h,1) = find(Q_state.o(:,1) == 1);
    end

    for t = 2:T
        % generative process
        Q_state.s(:,t) = mnrnd(1, B * calc_DxS(Q_state.u(:,t-1), Q_state.s(:,t-1)));
        Q_state.o(:,t) = mnrnd(1, A * Q_state.s(:,t));

        % inference
        Q_state.vs(:,t) = qlnA' * Q_state.o(:,t) + qlnB * calc_DxS(Q_state.u(:,t-1), Q_state.qs(:,t-1));
        Q_state.vd(:,t) = qlnC1 * Q_state.qs(:,t-1) + lnE;
        Q_state.qs(:,t) = exp(Q_state.vs(:,t)-max(Q_state.vs(:,t))) ./ sum(exp(Q_state.vs(:,t)-max(Q_state.vs(:,t))));
        Q_state.qd(:,t) = exp(Q_state.vd(:,t)-max(Q_state.vd(:,t))) ./ sum(exp(Q_state.vd(:,t)-max(Q_state.vd(:,t))));

        % decision
        Q_state.d(:,t) = mnrnd(1, Q_state.qd(:,t));
        Q_state.u(:,t) = kron(eye(cfg.Nu), ones(1, Nd/cfg.Nu)) * Q_state.d(:,t);

        % risk: 0 when the hand got closer to the target, 1 when it moved away
        nowMd = manhattan_distance_validation(Nm, Q_state.s(:,t), targetX, targetY);
        preMd = distance(t-1);
        distance(t) = nowMd;
        if preMd > nowMd
            Q_state.G(t) = 0;
        elseif preMd < nowMd
            Q_state.G(t) = 1;
        else
            Q_state.G(t) = 0.5;
        end

        if cfg.draw == 1
            figure_output_validation(Nm, Q_state.s(:,t), targetX, targetY, C4disp);
            title(['Trial:', num2str(h,'%03d'), ', Step:', num2str(t,'%03d')]);
            drawnow;
        end

        if cfg.record_traces == 1
            rslt.info_s(h,t) = find(Q_state.s(:,t) == 1);
            rslt.info_o(h,t) = find(Q_state.o(:,t) == 1);
        end

        % No early exit when the target is reached: every session contributes
        % exactly T trials of experience.
    end
    Q_state.t = t;

    % Risk over the partition of Manhattan-distance changes. The two learners
    % are given separate risk profiles so they can be set independently.
    risk_high     = 0.5 + cfg.risk_range;
    risk_low      = 0.5 - cfg.risk_range;
    risk_transfer = [risk_high, risk_high, risk_high, risk_high, risk_high, risk_low, risk_low];
    risk_naive    = risk_transfer;
    partition     = [-1, 0, 1];
    Q_state.Gamma_t = calc_risk(Q_state, cfg.range, partition, risk_transfer);
    Q_state.Gamma_n = calc_risk(Q_state, cfg.range, partition, risk_naive);

    %% Learning: lambda for the transfer learner, qcValid for the naive one
    qs = Q_state.qs(:, 1:t-1);
    d  = Q_state.d(:, 2:t);

    Lambda  = log(lambda / sum(lambda));
    vlambda = zeros(size(lambda));
    for i = 1:t-2
        G_t = Q_state.Gamma_t(i+2);
        G_n = Q_state.Gamma_n(i+2);
        for j = 1:Nlambda
            vlambda(j) = vlambda(j) + ((1-2*G_t) * preqlnC(:,:,j) * qs(:,i))' * d(:,i);
        end
        qcValid = max(qcValid + (1-2*G_n) * d(:,i) * qs(:,i)', 0);
    end

    vlambda = Lambda + vlambda;
    lambda  = exp(vlambda-max(vlambda)) ./ sum(exp(vlambda-max(vlambda)));

    lambdaList(h,:)  = lambda;
    vlambdaList(h,:) = vlambda;
    rslt.lambda(h,:) = lambda;
    rslt.qcList(h).qc = qcValid;
end

if cfg.saveMat == 1
    qcList = rslt.qcList;
    save(filename + ".mat", 'lambdaList', 'vlambdaList', 'qcList', 'cfg', '-v7.3');
end
end

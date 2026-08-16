%REPEAT_VALIDATION  Run the batch of 3x3 validation simulations and export them.
%   Runs VALIDATION_LEARNER once per simulation in cfg.simNums, saves the
%   collected results to cfg.rslt_File, then aggregates the learning curves and
%   writes them to cfg.xls_File.
%
%   Set cfg.aggregateOnly = 1 to skip the simulations and re-export an existing
%   cfg.rslt_File:
%       cfg = validation_config('aggregateOnly', 1, ...
%                               'rslt_File', "valid_calc10_qc0Ones_33_summary.mat");
%
%   Both learners are scored the same way: their C is reconstructed and compared
%   against the reference policy in cfg.optC_File through trace(dC * dC').
%     transfer  C = exp(sum_k lambda_k * log preqC_k)
%     naive     C = param_normalization(qcValid, "A")
%   The two lambda sheets additionally measure how close lambda itself is to the
%   optimal one-hot weighting, which only the transfer learner has.
%
%   Sheet layout (paper_figures.py depends on it):
%     A2 down   simulation number
%     B1 across session index 0 .. Nsession
%     B2 down   the session-0 baseline, constant across simulations
%     C2        the Nsim x Nsession matrix itself
%
%   MATLAB's current folder must be the repository root.
%
%   See also VALIDATION_CONFIG, VALIDATION_LEARNER, PREPARE_VALIDATION.

clear;

cfg = validation_config();

%% Run the batch ----------------------------------------------------------
if cfg.aggregateOnly
    loaded = load(cfg.rslt_File, "rslt");
    rslt   = loaded.rslt;
    fprintf('aggregateOnly: loaded %s\n', cfg.rslt_File);
else
    premsg = sprintf('progress: 0%%');
    fprintf('%s', premsg);
    for simN = cfg.simNums
        c      = cfg;
        c.simN = simN;
        c.seed = cfg.seedcore + 10000000 + simN;

        rslt(simN) = validation_learner(c); %#ok<SAGROW>

        msg = sprintf('progress: %d%%', simN);
        update_disp(premsg, msg, simN == cfg.simNums(end));
        premsg = msg;
        close all;
    end

    save(cfg.rslt_File, "rslt", '-v7.3');
    fprintf('wrote %s\n', cfg.rslt_File);
end

%% Reference quantities ---------------------------------------------------
learnedC = load_pretrained_policy(cfg);
Nlambda  = size(learnedC, 3);

loaded = load(cfg.optC_File, "optC");
optC   = loaded.optC;

Nm          = cfg.Nm;
Nd          = cfg.Nd;
Ns          = cfg.Ns;
targetIndex = cfg.targetIndex;

% Taken from the data rather than from cfg, so an older or partial batch
% aggregates correctly without having to match cfg.Nsim / cfg.Nsession by hand.
Nsim     = numel(rslt);
Nsession = size(rslt(1).lambda, 1);
fprintf('aggregating %d simulations x %d sessions\n', Nsim, Nsession);

% Optimal lambda puts all the weight on the pretrained policy for the target.
optLambda = zeros(1, Nlambda);
optLambda(cfg.targetXY) = 1;

preList = struct('qc', cell(1, Nlambda), 'qlnC_i', cell(1, Nlambda));
for i = 1:Nlambda
    [preList(i).qc, preList(i).qlnC_i] = param_normalization(learnedC(:,:,i), "A");
end

%% Error curves -----------------------------------------------------------
differenceProposed   = zeros(Nsim, Nsession);
differenceNaive      = zeros(Nsim, Nsession);
differenceLambda     = zeros(Nsim, Nsession);
differenceLambda_opt = zeros(Nsim, Nsession);

for i = 1:Nsim
    for j = 1:Nsession
        % transfer: rebuild C from the mixing weights
        qlnC = zeros(Nd, Ns);
        for k = 1:Nlambda
            qlnC = qlnC + rslt(i).lambda(j,k) * preList(k).qlnC_i;
        end
        dirCProposed = exp(qlnC);
        qcProposed = zeros(Nd, Ns);
        qcProposed(:,targetIndex) = dirCProposed(:,targetIndex);

        % naive: C is learned directly
        dirCNaive = param_normalization(rslt(i).qcList(j).qc, "A");
        qcNaive = zeros(Nd, Ns);
        qcNaive(:,targetIndex) = dirCNaive(:,targetIndex);

        difMatrixProposed = qcProposed - optC;
        differenceProposed(i,j) = trace(difMatrixProposed*difMatrixProposed');
        difMatrixNaive = qcNaive - optC;
        differenceNaive(i,j) = trace(difMatrixNaive*difMatrixNaive');

        difLambda = rslt(i).lambda(j,:) - optLambda;
        differenceLambda(i,j) = trace(difLambda*difLambda');
        difLambda_opt = rslt(i).lambda(j,cfg.targetXY) - 1;
        differenceLambda_opt(i,j) = trace(difLambda_opt*difLambda_opt');
    end
end

%% Session-0 baselines ----------------------------------------------------
dirC0 = param_normalization(rslt(1).qcValid0, "A");
qc0 = zeros(Nd, Ns);
qc0(:,targetIndex) = dirC0(:,targetIndex);
difMat0 = qc0 - optC;
dif0 = trace(difMat0*difMat0');

lambda0 = ones(1, Nlambda)/Nlambda;
dif0Lambda     = trace((lambda0-optLambda)*(lambda0-optLambda)');
dif0Lambda_opt = trace((lambda0(1,cfg.targetXY)-1)*(lambda0(1,cfg.targetXY)-1)');

%% Excel export -----------------------------------------------------------
x = 0:Nsession;
simLabels = (1:Nsim)';

write_sheet(cfg.xls_File, 'transfer', differenceProposed/Nd/(Nm^2), ...
            dif0/Nd/(Nm^2), simLabels, x);
write_sheet(cfg.xls_File, 'naive', differenceNaive/Nd/(Nm^2), ...
            dif0/Nd/(Nm^2), simLabels, x);
write_sheet(cfg.xls_File, 'lambda_Gerror', differenceLambda/Nlambda, ...
            dif0Lambda/Nlambda, simLabels, x);
write_sheet(cfg.xls_File, 'lambda_opt_Gerror', differenceLambda_opt/Nlambda, ...
            dif0Lambda_opt/Nlambda, simLabels, x);

fprintf('wrote %s\n', cfg.xls_File);


function write_sheet(file, sheet, data, baseline, simLabels, x)
%WRITE_SHEET  Write one learning curve plus its labels to an Excel sheet.
%   Column B holds session 0 (the constant baseline), C onwards sessions 1..N.

writematrix(baseline*ones(numel(simLabels),1), file, 'Sheet', sheet, 'Range', 'B2');
writematrix(data,      file, 'Sheet', sheet, 'Range', 'C2');
writematrix(x,         file, 'Sheet', sheet, 'Range', 'B1');
writematrix(simLabels, file, 'Sheet', sheet, 'Range', 'A2');
end

%REPEAT_MAIN  Run a batch of prism-adaptation simulations and export the results.
%   Runs MAIN_LEARNER once per simulation in cfg.simNums, saves the raw results
%   to a .mat file and writes the summary sheets to a .xlsx file.  By default
%   that is every simulation the input files hold, 1..cfg.Nsim.
%
%   Switch between the two learners with the sim_type override below; nothing
%   else needs to change.  Run PREPARE_ALL first to generate the input files.
%   MATLAB's current folder must be the repository root.
%
%   See also DEFAULT_CONFIG, MAIN_LEARNER, PREPARE_ALL.

clear;

cfg = default_config('sim_type', "transfer");

%% Run the batch ----------------------------------------------------------
premsg = sprintf('progress: 0%%');
fprintf('%s', premsg);
for simN = cfg.simNums
    c      = cfg;
    c.simN = simN;
    c.seed = cfg.seedcore + 10000000 + simN;

    rslt(simN) = main_learner(c); %#ok<SAGROW>

    msg = sprintf('progress: %d%%', simN);
    update_disp(premsg, msg, simN == cfg.simNums(end));
    premsg = msg;
    close all;
end

%% Output file names ------------------------------------------------------
% Both names come from the registry, so they carry every condition that went
% into the run.
mat_filename = cfg.rslt_File;
xls_filename = cfg.xls_File;

save(mat_filename, "rslt", '-v7.3');

%% Per-simulation target positions ----------------------------------------
% randhand 3 and 4 randomise the target per simulation, so the baseline and
% exposure targets are read back out of posSet rather than assumed.
tXbase = zeros(cfg.Nsim,1); tYbase = zeros(cfg.Nsim,1);
tXexpo = zeros(cfg.Nsim,1); tYexpo = zeros(cfg.Nsim,1);
if cfg.randhand >= 3
    loaded = load(cfg.InitPos_File, "posSet");
    posSet = loaded.posSet;
    for i = cfg.simNums
        tXbase(i) = posSet(i).targetX_Baseline;
        tYbase(i) = posSet(i).targetY_Baseline;
        tXexpo(i) = posSet(i).targetX_Exposure;
        tYexpo(i) = posSet(i).targetY_Exposure;
    end
else
    tXbase(cfg.simNums) = 7;
    tYbase(cfg.simNums) = 5;
    tXexpo(cfg.simNums) = 7 - cfg.dis;
    tYexpo(cfg.simNums) = 5;
end
% Column mask index of the baseline target, and the index of the pretrained
% policy that is optimal once the prism is on (the exposure target).
baselineIndex = (tXbase-1)*cfg.Nm + tYbase;
exposureIndex = (tXexpo-1)*cfg.Nm + tYexpo;

%% Hand position at the step where the shortest path would have finished ---
loaded = load(cfg.shortestDis_File);
if cfg.randhand == 2
    shortestDis = loaded.shortestDis_hand;
else
    shortestDis = loaded.shortestDis_pos;
end

% Oriented as (session, sim), matching both the preallocation and shortestDis.
posPastShortestStep = zeros(cfg.Nsession, cfg.Nsim);
for i = cfg.simNums
    for j = 1:cfg.Nsession
        posPastShortestStep(j,i) = rslt(i).info_s(j, shortestDis(j,i)+1);
    end
end

handPos   = floor(posPastShortestStep / cfg.Nm^2);
targetPos = mod(posPastShortestStep, cfg.Nm^2);
handX     = floor(handPos / cfg.Nm) + 1;
targetX   = floor(targetPos / cfg.Nm) + 1;

% Transposed on the way out: the sheets are laid out as (sim, session).
absPos = handX';
relPos = (handX - targetX)';

write_sheet(xls_filename, 'absPos', absPos(cfg.simNums,:), cfg.simNums, 1:cfg.Nsession);
write_sheet(xls_filename, 'relPos', relPos(cfg.simNums,:), cfg.simNums, 1:cfg.Nsession);

%% Lambda trajectories (transfer learner only) ----------------------------
if cfg.sim_type == "transfer"
    truelambdas    = zeros(cfg.Nsim, cfg.Nsession);
    shiftedlambdas = zeros(cfg.Nsim, cfg.Nsession);
    for i = cfg.simNums
        truelambdas(i,:)    = rslt(i).lambda(:, exposureIndex(i))';
        shiftedlambdas(i,:) = rslt(i).lambda(:, baselineIndex(i))';
    end
    write_sheet(xls_filename, 'truelambdaList',    truelambdas(cfg.simNums,:),    cfg.simNums, 1:cfg.Nsession);
    write_sheet(xls_filename, 'shiftedlambdaList', shiftedlambdas(cfg.simNums,:), cfg.simNums, 1:cfg.Nsession);
end

%% Trials needed per session ----------------------------------------------
tList = zeros(cfg.Nsim, cfg.Nsession);
for i = cfg.simNums
    tList(i,:) = rslt(i).tList;
end
write_sheet(xls_filename, 't', tList(cfg.simNums,:), cfg.simNums, 1:cfg.Nsession);

%% Generalisation error ---------------------------------------------------
% Measured over the exposure period against this simulation's own optimum:
%   C      the optimal policy for its perceived/real target pair (MAKE_OPTC)
%   lambda all the weight on the pretrained policy for its exposure target
errorGraph_index = (cfg.exposureStart-1):(cfg.removalStart-1);
interval = cfg.Nm^2;

% Same stack, blurred the same way, that the learner acted on; loading it raw
% here would make the reconstructed C disagree with the simulated one.
learnedC = load_pretrained_policy(cfg);
Nlambda  = size(learnedC, 3);
if cfg.sim_type == "transfer"
    for k = 1:Nlambda
        [~, preList(k).qlnC_i] = param_normalization(learnedC(:,:,k), "A"); %#ok<SAGROW>
    end
end

difference           = zeros(cfg.Nsim, cfg.Nsession);
differenceLambda     = zeros(cfg.Nsim, cfg.Nsession);
differenceLambda_opt = zeros(cfg.Nsim, cfg.Nsession);

for i = cfg.simNums
    % All states whose target part equals this simulation's baseline target.
    targetIndex = baselineIndex(i):interval:cfg.Ns;
    optC = make_optC(cfg.Nm, tXbase(i), tYbase(i), tXexpo(i), tYexpo(i));
    optLambda = zeros(1, Nlambda);
    optLambda(exposureIndex(i)) = 1;

    for j = errorGraph_index
        switch cfg.sim_type
            case "naive"
                dirC = param_normalization(rslt(i).qcList(j).qc, "A");
            case "transfer"
                qlnC = zeros(cfg.Nd, cfg.Ns);
                for k = 1:Nlambda
                    qlnC = qlnC + rslt(i).lambda(j,k) * preList(k).qlnC_i;
                end
                dirC = exp(qlnC);
        end
        qc = zeros(cfg.Nd, cfg.Ns);
        qc(:,targetIndex) = dirC(:,targetIndex);

        difMatrix = qc - optC;
        difference(i,j) = trace(difMatrix*difMatrix');

        if cfg.sim_type == "transfer"
            difLambda = rslt(i).lambda(j,:) - optLambda;
            differenceLambda(i,j) = trace(difLambda*difLambda');
            difLambda_opt = rslt(i).lambda(j, exposureIndex(i)) - 1;
            differenceLambda_opt(i,j) = trace(difLambda_opt*difLambda_opt');
        end
    end
end

write_sheet(xls_filename, 'C_Gerror', ...
            difference(cfg.simNums, errorGraph_index)/cfg.Nd/(cfg.Nm^2), ...
            cfg.simNums, errorGraph_index);

if cfg.sim_type == "transfer"
    write_sheet(xls_filename, 'lambda_Gerror', ...
                differenceLambda(cfg.simNums, errorGraph_index)/Nlambda, ...
                cfg.simNums, errorGraph_index);
    write_sheet(xls_filename, 'lambda_opt_Gerror', ...
                differenceLambda_opt(cfg.simNums, errorGraph_index)/Nlambda, ...
                cfg.simNums, errorGraph_index);
end


function write_sheet(file, sheet, data, rowLabels, colLabels)
%WRITE_SHEET  Write one matrix and its row/column labels to an Excel sheet.
%   Rows are simulations, columns are sessions.

writematrix(data,         file, 'Sheet', sheet, 'Range', 'B2');
writematrix(colLabels(:)', file, 'Sheet', sheet, 'Range', 'B1');
writematrix(rowLabels(:),  file, 'Sheet', sheet, 'Range', 'A2');
end

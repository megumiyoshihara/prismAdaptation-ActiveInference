%REWRITE_EXCEL  Re-export the Excel sheets from a Result_*.mat saved earlier.
%   REPEAT_MAIN writes both the .mat and the .xlsx in one go.  This script
%   redoes only the .xlsx half, from a .mat that is already on disk, so a sheet
%   can be regenerated without spending hours re-simulating.
%
%   It is a script, not a function: describe the run in the DEFAULT_CONFIG call
%   below, then run it from the repository root.  The registry turns that
%   description into the .mat to read, the .xlsx to write and the matching
%   shortest-distance file, so the three cannot disagree.  What matters is
%   sim_type, randhand, MD, blur, risk_divisor and simNums -- the same settings
%   the run itself was made with.
%
%   The sheets written are those paper_figures.py reads: absPos, relPos, t,
%   C_Gerror, and for the transfer learner truelambdaList, shiftedlambdaList,
%   lambda_Gerror and lambda_opt_Gerror.  Every sheet has session numbers in
%   row 1 and simulation numbers in column A.
%
%   The computations are copies of the corresponding sections of REPEAT_MAIN,
%   with the parameters spelled out as literals instead of coming from
%   DEFAULT_CONFIG, so keep the two in step when either changes.
%
%   See also REPEAT_MAIN, DEFAULT_CONFIG.

clear

% Describe the run whose .mat is being re-exported.  The registry then resolves
% every file name from it, exactly as REPEAT_MAIN does.
cfg = default_config( ...
    'sim_type',     "transfer", ...
    'randhand',     2, ...
    'risk_divisor', 100, ...
    'simNums',      1:100);

loaded = load(cfg.rslt_File, "rslt");
rslt   = loaded.rslt;
xls_filename = cfg.xls_File;

sim_type = double(cfg.sim_type == "transfer");
randhand = cfg.randhand;
Nsim     = cfg.Nsim;
Nsession = cfg.Nsession;
Nm       = cfg.Nm;
Ns       = cfg.Ns;
Nd       = cfg.Nd;
Nlambda  = Nm^2;

simNums = 1:Nsim;

% Initialize the result matrix for storing positions
posPastShortestStep = zeros(Nsim,Nsession);

loaded = load(cfg.shortestDis_File);
if randhand == 2
    shortestDis = loaded.shortestDis_hand;
else
    shortestDis = loaded.shortestDis_pos;
end


for i=simNums
    for j = 1:Nsession
        Ntrial = rslt(i).tList(:,j);
        posPastShortestStep(i,j) = rslt(i).info_s(j,shortestDis(j,i)+1);

    end
end

handPos = floor(posPastShortestStep/100);
targetPos = mod(posPastShortestStep,100);
handX = floor(handPos/10)+1;
targetX = floor(targetPos/10)+1;

absPos = handX;
relPos = handX-targetX;



%%%%%%%%%%%%%%%%%%%%%%%%%%%
if sim_type == 1
    baselineStartIndex = 1;
    exposureStartIndex = 141;
    targetXList_baseline = zeros(Nsim,1);
    targetYList_baseline = zeros(Nsim,1);
    targetXList_exposure = zeros(Nsim,1);
    targetYList_exposure = zeros(Nsim,1);
    for i = simNums
        targetXList_baseline(i) = floor(mod(rslt(i).info_s(baselineStartIndex),100)/10)+1;
        targetYList_baseline(i) = floor(mod(rslt(i).info_s(baselineStartIndex),10));
        targetXList_exposure(i) = floor(mod(rslt(i).info_s(exposureStartIndex),100)/10)+1;
        targetYList_exposure(i) = floor(mod(rslt(i).info_s(exposureStartIndex),10));
        for j = 1:Nsession
            lambdaList(i,j).lambda = rslt(i).lambda(j,:);
        end
    end
    baselineIndex = (targetXList_baseline-1)*10+targetYList_baseline;
    exposureIndex = (targetXList_exposure-1)*10+targetYList_exposure;
    
    for i=simNums
        for j=1:Nsession
            truelambdas(i,j) = lambdaList(i,j).lambda(:,exposureIndex(i));
            shiftedlambdas(i,j) = lambdaList(i,j).lambda(:,baselineIndex(i));
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculation Generarization error
exposureStart = 141;
removalStart = 171;
errorGraph_index = exposureStart-1:removalStart-1;
interval = 100;
%%{

load(cfg.reference_policy_File, "optC");
difference = zeros(Nsim,Nsession);
if sim_type == 1
    learnedC = load_pretrained_policy(cfg);
    for i = 1:Nlambda
        [preList(i).qc, preList(i).qlnC_i] = param_normalization(learnedC(:,:,i), "A");
    end
end
%index = [1,10:10:Nsession];

for i = simNums
    tList(:,i) = rslt(i).tList';
    targetXY = mod(rslt(i).info_s(1,1),100);
    targetIndex = targetXY:interval:Ns;

    optLambda = zeros(1,Nlambda);
    optLambda(1,mod(rslt(i).info_s(exposureStart,1),100)) = 1;
    
    for j= errorGraph_index
        %%{
        qlnC = zeros(Nd,Ns);
        if sim_type == 0
            dirC = param_normalization(rslt(i).qcList(j).qc, "A");
            qc = zeros(Nd,Ns);
            qc(:,targetIndex) = dirC(:,targetIndex);
            
            difMatrix = qc-optC;
            difference(i,j) = trace(difMatrix*difMatrix');
            
        elseif sim_type == 1
            for k=1:Nlambda
                qlnC = qlnC + rslt(i).lambda(j,k) * preList(k).qlnC_i;
            end
            dirC = exp(qlnC);
            qc = zeros(Nd,Ns);
            qc(:,targetIndex) = dirC(:,targetIndex);
            
            difMatrix = qc-optC;
            difference(i,j) = trace(difMatrix*difMatrix');
            
            difLambda = rslt(i).lambda(j,:)-optLambda;
            differenceLambda(i,j) = trace(difLambda*difLambda');
            difLambda_opt = rslt(i).lambda(j,35)-1;
            differenceLambda_opt(i,j) = trace(difLambda_opt*difLambda_opt');
        end

    end
    
end
%}

writematrix(absPos,xls_filename,'Sheet','absPos','Range','B2');
writematrix(1:Nsession,xls_filename,'Sheet','absPos','Range','B1');
writematrix((simNums)',xls_filename,'Sheet','absPos','Range','A2');

writematrix(relPos,xls_filename,'Sheet','relPos','Range','B2');
writematrix(1:Nsession,xls_filename,'Sheet','relPos','Range','B1');
writematrix((simNums)',xls_filename,'Sheet','relPos','Range','A2');

if sim_type == 1
    writematrix(truelambdas,xls_filename,'Sheet','truelambdaList','Range','B2');
    writematrix(1:Nsession,xls_filename,'Sheet','truelambdaList','Range','B1');
    writematrix((simNums)',xls_filename,'Sheet','truelambdaList','Range','A2');
    
    writematrix(shiftedlambdas,xls_filename,'Sheet','shiftedlambdaList','Range','B2');
    writematrix(1:Nsession,xls_filename,'Sheet','shiftedlambdaList','Range','B1');
    writematrix((simNums)',xls_filename,'Sheet','shiftedlambdaList','Range','A2');
end

writematrix(tList',xls_filename,'Sheet','t','Range','B2');
writematrix(1:Nsession,xls_filename,'Sheet','t','Range','B1');
writematrix((simNums)',xls_filename,'Sheet','t','Range','A2');
%%{
writematrix(difference(:,errorGraph_index)/4/(Nm^2),xls_filename,'Sheet','C_Gerror','Range','B2');
writematrix(errorGraph_index,xls_filename,'Sheet','C_Gerror','Range','B1');
writematrix((simNums)',xls_filename,'Sheet','C_Gerror','Range','A2');

if sim_type == 1
    writematrix(differenceLambda(:,errorGraph_index)/Nlambda,xls_filename,'Sheet','lambda_Gerror','Range','B2');
    writematrix(errorGraph_index,xls_filename,'Sheet','lambda_Gerror','Range','B1');
    writematrix((simNums)',xls_filename,'Sheet','lambda_Gerror','Range','A2');
    
    writematrix(differenceLambda_opt(:,errorGraph_index)/Nlambda,xls_filename,'Sheet','lambda_opt_Gerror','Range','B2');
    writematrix(errorGraph_index,xls_filename,'Sheet','lambda_opt_Gerror','Range','B1');
    writematrix((simNums)',xls_filename,'Sheet','lambda_opt_Gerror','Range','A2');
end
%}
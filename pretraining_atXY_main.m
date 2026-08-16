function preQ = pretraining_atXY_main(Nm, T,w_section,seed, targetX, targetY, A_step)
%PRETRAINING_ATXY_MAIN  Learn one main-experiment policy for a fixed target.
%   PREQ = PRETRAINING_ATXY_MAIN(NM,T,W_SECTION,SEED,TARGETX,TARGETY,A_STEP)
%   runs T trials of naive learning against the target (TARGETX,TARGETY), in
%   sections of W_SECTION trials with the hand re-drawn at random each section,
%   and returns the learned qa / qb / qc together with the settings used.
%   A_STEP is the blur of the initial likelihood (see MAKE_INIT_A_FOR_PRETRAIN).
%
%   The result is also saved under pretrain/, which is created if missing.
%   PREPARE_PRETRAINING calls this once per target and stacks the policies.
%
%   See also PREPARE_PRETRAINING, PRETRAINING_ATXY_VALIDATION, NAIVE_LEARNING.
range = 1; % learning partition


% Write-only archive of the individual pretraining runs; PREPARE_PRETRAINING
% uses the returned preQ and nothing reads these files back, so the name is not
% in the DATA_FILES registry.
Name_folder = "pretrain/";
Name_init ="policy_main";
Name_NumT = "T" + num2str(T);
Name_Seed = "seed" + num2str(seed+10000000);
filename = Name_folder + Name_init + "_" + "Ttgt"+num2str(targetX)+num2str(targetY)+"_" + Name_NumT + "_" + Name_Seed;
% Check if the "Name_folder" folder exists, if not, create it
if ~exist(Name_folder, 'dir')
    mkdir(Name_folder);
end

Ns = Nm^2*Nm^2; % dimension of the hidden state
No = Ns; % dimensionality of the sensory input o
Nu = 4; % size of decision
Nd = Nu; % dimensionality of the decisions

preQ.Nm = Nm;
preQ.Ns = Ns;
preQ.No = No;
preQ.Nu = Nu;
preQ.Nd = Nd;

qaWeight = 100^(-1);
qbWeight = 100^(-1);
qcWeight = 10^(0);
A = eye(No, Ns, 'single');
B_target = eye(Nm^2, Nm^2);
B_hand = make_Bhand(Nm);
B = kron(B_hand , B_target);
D        = ones(Ns,1,'single')*0.5/Ns;
E        = ones(Nd,1,'single')*0.5/Nd;

qb0     = ones(size(B),'single') * qbWeight;
qc0     = ones(Nd,Ns,'single') * qcWeight;

qa0 = A_step;
prior = 0.25; % prior of action

preQ.qaWeight = qaWeight;
preQ.qbWeight = qbWeight;
preQ.qcWeight = qcWeight;
preQ.prior = prior;
preQ.T = T;
preQ.qa0 = qa0;
preQ.qb0 = qb0;
preQ.qc0 = qc0;
E = kron([prior, prior, prior, prior]',ones(Nd/4,1)/(Nd/4));


rng(seed+10000000);
preQ.seed = seed+10000000;

distance = zeros(w_section,1);

S = fix(T/w_section);

preQ.risk = [1, 1, 1, 1, 1, 0, 0];
preQ.partition = [-1,0,1];
Q_state =  naive_init(qa0, qb0, qc0, D, E, w_section);
Q_update =  naive_init(qa0, qb0, qc0, D, E, w_section);
qA = qa0/sum(qa0);
qlnA = log(qa0);
[qb, qlnB] = param_normalization(Q_state.qb, "A");
[qc, qlnC] = param_normalization(Q_state.qc, "A");
lnD = log(max(10^-6, Q_state.D));
lnE = log(max(10^-6, Q_state.E));

for s=1:S
    Q_state =  naive_init(qa0, qb0, qc0, D, E, w_section);
    
    Q_state.s(:, 1)=make_randHandS(Nm, targetX, targetY); % random hand init
    distance(1,1)=manhattan_distance(Nm,Q_state.s(:,1));
    Q_state.u(1,1)=1;
    Q_state.d(1,1)=1;
    for t=2:w_section
        % generative process
            
        Q_state.s(:,t) = mnrnd(1, B * calc_DxS(Q_state.u(:,t-1), Q_state.s(:,t-1)));
        Q_state.o(:,t) = mnrnd(1, A *Q_state.s(:,t));
    
        % inference
        Q_state.vs(:,t)= qlnA' * Q_state.o(:,t)+qlnB * calc_DxS(Q_state.u(:, t-1), Q_state.qs(:, t-1));
        Q_state.vd(:,t)= qlnC * Q_state.qs(:,t-1)+lnE;
        Q_state.qs(:,t) = exp(Q_state.vs(:,t)-max(Q_state.vs(:,t)))./sum(exp(Q_state.vs(:, t)-max(Q_state.vs(:,t))));
        Q_state.qd(:,t) = exp(Q_state.vd(:,t)-max(Q_state.vd(:,t)))./sum(exp(Q_state.vd(:, t)-max(Q_state.vd(:,t))));
    
        % decision
        Q_state.d(:,t)= mnrnd(1, Q_state.qd(:,t));
        Q_state.u(:,t)=kron(eye(4),ones(1,Nd/4)) * Q_state.d(:,t);
    
        % risk
        nowMd = manhattan_distance(Nm,Q_state.s(:,t));
        preMD = distance(t-1,1);
        distance(t,1)= nowMd;
        if preMD > nowMd Q_state.G(t) = 0;
        elseif preMD < nowMd Q_state.G(t) = 1;
        else Q_state.G(t) = 0.5;
        end
    
        % figure output
        %figure_output_main(Nm, Q_state.s(:,t));
        %title(['pre training(',num2str(targetX),',',num2str(targetY),'): ', num2str((s-1)*w_section+t), '/', num2str(T)]);
        %drawnow;
        
        
    end
    Q_state.t =t;
    
    % compute risk
    
    Q_state.Gamma = calc_risk(Q_state, range, preQ.partition, preQ.risk);
    
    qa_ = Q_update.qa;
    qb_ = Q_update.qb;
    qc_ = Q_update.qc;
    Q_update = Q_state;
    Q_update.qa = qa_;
    Q_update.qb = qb_;
    Q_update.qc = qc_;
    
    Q_update = naive_learning(Q_update);
end
preQ.qa = Q_update.qa;
preQ.qb = Q_update.qb;
preQ.qc = Q_update.qc;

save(filename+".mat", "preQ",'-v7.3');
end

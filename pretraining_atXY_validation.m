function preQ = pretraining_atXY_validation(Nm,T,w_section,seed, targetX, targetY, A_step)
%PRETRAINING_ATXY_VALIDATION  Learn one validation policy for a fixed target.
%   PREQ = PRETRAINING_ATXY_VALIDATION(NM,T,W_SECTION,SEED,TARGETX,TARGETY,A_STEP)
%   is the validation-side counterpart of PRETRAINING_ATXY_MAIN: the state holds
%   the hand only (Ns = Nm^2), so the target enters through the arguments.  It
%   runs T trials in sections of W_SECTION and returns the learned qa / qb / qc.
%
%   The result is also saved under pretrain/, which is created if missing.
%
%   See also PREPARE_VALIDATION, PRETRAINING_ATXY_MAIN, NAIVE_LEARNING.

range = 1; % learning partition


% Write-only archive, as in PRETRAINING_ATXY_MAIN.
Name_folder = "pretrain/";
Name_init ="policy_valid";
Name_NumT = "T" + num2str(T);
Name_Seed = "seed" + num2str(seed+10000000);
filename = Name_folder + Name_init + "_" + "Ttgt"+num2str(targetX)+num2str(targetY)+"_" + Name_NumT + "_" + Name_Seed;
% Check if the "Name_folder" folder exists, if not, create it
if ~exist(Name_folder, 'dir')
    mkdir(Name_folder);
end

Ns = Nm^2; % dimension of the hidden state
No = Ns; % dimensionality of the sensory input o
Nu = 4; % size of decision
Nd = Nu; % dimensionality of the decisions

preQ.Nm = Nm;
preQ.Ns = Ns;
preQ.No = No;
preQ.Nu = Nu;
preQ.Nd = Nd;

qaWeight = 100^(1);
qbWeight = 100^(1);
qcWeight = 10^(1);
A = eye(No, Ns, 'single');
B_hand = make_Bhand(Nm);
B = B_hand;
D        = ones(Ns,1,'single')*0.5/Ns;
E        = ones(Nd,1,'single')*0.5/Nd;

qa0 = A_step;
qb0     = ones(size(B),'single') * qbWeight;
qc0     = ones(Nd,Ns,'single') * qcWeight;

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
[qA, qlnA] = param_normalization(Q_state.qa, "A");
[qb, qlnB] = param_normalization(Q_state.qb, "A");
[qc, qlnC] = param_normalization(Q_state.qc, "A");
lnD = log(max(10^-6, Q_state.D));
lnE = log(max(10^-6, Q_state.E));

for s=1:S
    Q_state =  naive_init(qa0, qb0, qc0, D, E, w_section);
    
    Q_state.s(:, 1)=make_randHandS_validation(Nm,targetX,targetY); % random hand init
    distance(1)=manhattan_distance_validation(Nm,Q_state.s(:,1),targetX,targetY);
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
        nowMd = manhattan_distance_validation(Nm,Q_state.s(:,t),targetX,targetY);
        preMD = distance(t-1);
        distance(t)= nowMd;
        if preMD > nowMd Q_state.G(t) = 0;
        elseif preMD < nowMd Q_state.G(t) = 1;
        else Q_state.G(t) = 0.5;
        end
    
        % figure output
        %figure_output_validation(Nm, Q_state.s(:,t),targetX,targetY);
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

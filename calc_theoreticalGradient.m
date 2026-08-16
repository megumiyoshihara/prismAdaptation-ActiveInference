%CALC_THEORETICALGRADIENT  Gradient separating the best two pretrained policies.
%   Scores every pretrained goal prior of the 3x3 validation experiment against
%   the analytical optimum X_star, and reports twice the gap between the best
%   and the second-best score.  That gap is the slope with which the mixing
%   weight lambda of the correct policy grows, i.e. the analytical prediction
%   the measured lambda curve is compared against.
%
%   For each pretrained policy i the score is
%       lncx(i) = risk_weight * <log C_i, X_star>
%   and the reported value is grad = 2*(largest - second largest).
%
%   Run as a script from the repository root.  Nothing is saved; read grad from
%   the workspace afterwards.  Inputs come from VALIDATION_CONFIG:
%     cfg.pretrained_policy_File   learnedC, the pretrained policies (4 x 9 x 9)
%     cfg.optC_File                X_star, the 3x3 analytical optimum (4 x 9)
%
%   See also PREPARE_VALIDATION, MAKE_OPTC.

clear
cfg = validation_config();
learnedC = load_pretrained_policy(cfg);
load(cfg.optC_File, "X_star");
Nm = 3;
Nlambda = 9;
Nc_vec = 4*Nlambda;
risk_weight = 0.1;

qlnC = zeros(Nlambda,Nc_vec);

for i=1:Nlambda
    preQ(i).qc = learnedC(:,:,i); % Assuming computeLambda is a user-defined function
    [preQ(i).qC,~] = param_normalization(preQ(i).qc,"A");
    preQ(i).qlnC = log(preQ(i).qC);
    qlnC(i,:) = reshape(preQ(i).qlnC,1,Nc_vec);
end
X_star_vec = reshape(X_star,Nc_vec,1);

lncx = qlnC*X_star_vec*risk_weight;

optC_vec_sorted = sort(lncx,'descend');


grad = 2*(optC_vec_sorted(1)-optC_vec_sorted(2));


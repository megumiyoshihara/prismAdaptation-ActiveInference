function init_A = make_init_A_for_pretrain(Nm, Nstep)
%MAKE_INIT_A_FOR_PRETRAIN  Blurred likelihood A that seeds the pretraining.
%   INIT_A = MAKE_INIT_A_FOR_PRETRAIN(NM,NSTEP) spreads every state's mass over
%   the cells reachable within NSTEP grid steps, then normalises the columns.
%   A larger NSTEP starts the pretraining from a vaguer sensory model.
%
%   See also PREPARE_PRETRAINING.
arguments (Input)
    Nm = 10;
    Nstep = 4;
end

arguments (Output)
    init_A;
end


Ns_ht = Nm^2;
Ns =Nm^2*Nm^2;
Nbase = 2*Nm-1;

% Calculate the range of movement possible with one STEP.
function after = oneStepAfter(size, before)
    M_wid = [zeros(size,1, 'single'),[eye(size-1, 'single');zeros(1,size-1, 'single')]];
    A_up = M_wid;
    A_down = flip(flip(M_wid,1),2);
    A_right = M_wid;
    A_left = flip(flip(M_wid,1),2);
    after = sign(before+A_up*before+A_down*before+before*A_right+before*A_left);
end

% Calculate the range of movement possible with N STEP.
function after = nStepAfter(size,before,n)
    if n == 1
        after = oneStepAfter(size,before);
    elseif n > 1
        m=floor(n/2);
        after = nStepAfter(size, nStepAfter(size, before, n-m),m);
    end
end


first = zeros(Nm,Nm);
first(Nm,Nm) = 1;
init = nStepAfter(Nm,first,Nstep);



A_Exinit = zeros(Ns,Ns,'single');
s_Handbases = eye(Ns_ht,Ns_ht);
s_Targetbases = zeros(Ns_ht,Ns_ht);
spread = zeros(Nbase,Nbase);
spread(1:Nm,1:Nm) = init;
spread(1:Nm,Nm+1:Nbase) = init(:,Nm-1:-1:1);
spread(Nm+1:Nbase,1:Nm) = init(Nm-1:-1:1,:);
spread(Nm+1:Nbase,Nm+1:Nbase) = init(Nm-1:-1:1,Nm-1:-1:1);

for i= 0:Nm-1
    for j = 0:Nm-1
            s_sample = reshape(spread(Nm-j:Nbase-j,Nm-i:Nbase-i), Ns_ht,1);
        for k=0:Ns_ht-1
            column_index = Nm*i+j+1;
            A_Exinit(:,Nm^2*k+Nm*i+j+1)=kron(s_Handbases(:,k+1),s_sample);
        
        end
    end
end

sumA = sum(A_Exinit);
maxA = max(sumA);


% Normalization
additionalTerm = 1;
base = maxA+additionalTerm-sumA;
newA = (1-A_Exinit).*(base./(Ns-sumA));
A_Ex = newA + A_Exinit;
init_A = A_Ex;
end
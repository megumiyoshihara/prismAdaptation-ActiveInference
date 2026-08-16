function Hand = make_initHand(Nsim,Nsession)
%MAKE_INITHAND  Hand positions per session for the fixed-target experiment.
%   HAND = MAKE_INITHAND(NSIM,NSESSION) draws, for every simulation and session,
%   a hand cell sharing neither row nor column with the current target, and packs
%   it as (handX-1)*Nm^3 + (handY-1)*Nm^2 + (targetX-1)*Nm + targetY.  The target
%   is (7,5) at baseline and after removal and (4,5) while the prism is on
%   (sessions 141..170).  Nm, the shift and the seed are fixed inside.
%
%   See also PREPARE_POSITIONS, MAKE_INITPOS_MD.
arguments (Input)
    Nsim
    Nsession
end

arguments (Output)
    Hand
end
seed = 63217438;
rng(seed);

Nm = 10; % size of monitor

dis = 3;
dir = "right";


targetX_baseline = 7;
targetY_baseline = 5;

if dir == "right"
    targetX_exposure = max(targetX_baseline-3,1);
    targetY_exposure = targetY_baseline;
elseif dir == "left"
    targetX_exposure = min(targetX_baseline+3,10);
    targetY_exposure = targetY_baseline;
end

for i = 1:Nsim
    for h=1:Nsession
    if h == 1
        targetX = targetX_baseline;
        targetY = targetY_baseline;
    elseif h == 141
        targetX = targetX_exposure;
        targetY = targetY_exposure;
    elseif h == 171
        targetX = targetX_baseline;
        targetY = targetY_baseline;
    end

    xlist = cat(2, 1:targetX-1, targetX+1:Nm);
    ylist = cat(2, 1:targetY-1, targetY+1:Nm);
    Xvec = zeros(Nm-1,1);
    Yvec = zeros(Nm-1,1);
    Xvec(randi(Nm-1, 1),1)=1;
    Yvec(randi(Nm-1, 1),1)=1;

    handX = xlist *Xvec;
    handY = ylist *Yvec;

    Hand.s(i,h)=(handX-1)*Nm^3+(handY-1)*Nm^2+(targetX-1)*Nm+targetY;
    end
end
end
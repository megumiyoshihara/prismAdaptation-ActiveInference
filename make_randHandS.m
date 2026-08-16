function s = make_randHandS(Nm, targetX, targetY)
%MAKE_RANDHANDS  Random hand position for a target, as a packed state vector.
%   S = MAKE_RANDHANDS(NM,TARGETX,TARGETY) draws a hand cell that shares neither
%   the row nor the column of the target, and packs the pair through CALC_XY2S.
%   The default target (Nm+1,Nm+1) is off the monitor, so the hand may then land
%   anywhere on the grid.
%
%   See also CALC_XY2S, MAKE_RANDHANDS_VALIDATION.
arguments
    Nm = 10;
    targetX = Nm+1;
    targetY = Nm+1;
end
xlist = cat(2, 1:targetX-1, targetX+1:Nm);
ylist = cat(2, 1:targetY-1, targetY+1:Nm);
Xvec = zeros(Nm-1,1);
Yvec = zeros(Nm-1,1);
Xvec(randi(Nm-1, 1),1)=1;
Yvec(randi(Nm-1, 1),1)=1;

x = xlist *Xvec;
y = ylist *Yvec;
s=calc_xy2S(Nm, targetX, targetY, x, y);

end
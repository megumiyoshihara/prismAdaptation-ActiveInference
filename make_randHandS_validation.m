function s = make_randHandS_validation(Nm, targetX, targetY)
%MAKE_RANDHANDS_VALIDATION  Random hand position as a validation state vector.
%   S = MAKE_RANDHANDS_VALIDATION(NM,TARGETX,TARGETY) draws a hand cell that
%   shares neither the row nor the column of the target and returns it one-hot as
%   (handX-1)*Nm + handY.  The validation state holds the hand only.
%
%   See also MAKE_RANDHANDS.
xlist = cat(2, 1:targetX-1, targetX+1:Nm);
ylist = cat(2, 1:targetY-1, targetY+1:Nm);
Xvec = zeros(Nm-1,1);
Yvec = zeros(Nm-1,1);
Xvec(randi(Nm-1, 1),1)=1;
Yvec(randi(Nm-1, 1),1)=1;

x = xlist *Xvec;
y = ylist *Yvec;
s= zeros(Nm^2,1);
s((x-1)*Nm+y)=1;

end
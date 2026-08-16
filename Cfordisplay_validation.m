function [C4disp] = Cfordisplay_validation(Nm,qc)
%CFORDISPLAY_VALIDATION  Policy laid back onto the monitor grid for display.
%   C4DISP = CFORDISPLAY_VALIDATION(NM,QC) averages each cell's policy over the
%   four neighbours its actions lead to, so the policy can be drawn on the
%   Nm x Nm grid.  Edge and corner cells use their reduced neighbourhoods.
%
%   See also CFORDISPLAY, FIGURE_OUTPUT_VALIDATION.

% Calculations are performed separately for the cases of the corners, edges, and the inside.

% Corners
zeromatrix = zeros(1,Nm^2);
tm = zeromatrix;
tm(Nm+1:Nm:Nm*(Nm-2)+1)=1;
topmost = tm;
bm = zeromatrix;
bm(2*Nm:Nm:Nm*(Nm-1))=1;
bottommost = bm;
lm = zeromatrix;
lm(2:1:Nm-1)=1;
leftmost = lm;
rm = zeromatrix;
rm(Nm^2-Nm+2:1:Nm^2-1)=1;
rightmost = rm;

% Edges
tl = zeromatrix;
tl(1) = 1;
topleft =tl;
tr = zeromatrix;
tr(Nm^2-Nm+1) = 1;
topright=tr;
bl = zeromatrix;
bl(Nm) = 1;
bottomleft=bl;
br = zeromatrix;
br(Nm^2) = 1;
bottomright=br;

% Inside
is = ones(1,Nm^2);
is = is-tm-bm-lm-rm-tl-tr-bl-br;
inside = is;

insideC=(circshift(qc,[0,1])+circshift(qc,[0,Nm])+circshift(qc,[0,Nm^2-1])+circshift(qc,[0,Nm^2-Nm]))*0.25;
insideC = insideC.*inside;

% Combine contributions from all regions to form the final policy display
topmostC = (circshift(qc,[0,1])+circshift(qc,[0,Nm])+qc+circshift(qc,[0,Nm^2-Nm]))*0.25;
topmostC = topmostC .*topmost;
bottommostC = (qc+circshift(qc,[0,Nm])+circshift(qc,[0,Nm^2-1])+circshift(qc,[0,Nm^2-Nm]))*0.25;
bottommostC = bottommostC .*bottommost;
leftmostC =(circshift(qc,[0,1])+circshift(qc,[0,Nm^1])+circshift(qc,[0,Nm^2-1])+qc)*0.25;
leftmostC = leftmostC .*leftmost;
rightmostC =(circshift(qc,[0,1])+qc+circshift(qc,[0,Nm^2-1])+circshift(qc,[0,Nm^2-Nm]))*0.25;
rightmostC = rightmostC .*rightmost;

topleftC = 0.5 * qc + 0.25 *( circshift(qc,[0,1]) + circshift(qc,[0,Nm^1]) );
topleftC = topleft .* topleftC;
toprightC =0.5 * qc + 0.25 *( circshift(qc,[0,1]) + circshift(qc,[0,Nm^2 - Nm]) );
toprightC = topright .* toprightC;
bottomleftC = 0.5 * qc + 0.25 *( circshift(qc,[0,Nm^2 - 1]) + circshift(qc,[0,Nm^1]) );
bottomleftC = bottomleft .* bottomleftC;
bottomrightC = 0.5 * qc + 0.25 *( circshift(qc,[0,Nm^2 - 1]) + circshift(qc,[0,Nm^2 - Nm]) );
bottomrightC = bottomright .* bottomrightC;

C4disp = insideC + topmostC + bottommostC + leftmostC + rightmostC + topleftC + toprightC + bottomleftC + bottomrightC;
end
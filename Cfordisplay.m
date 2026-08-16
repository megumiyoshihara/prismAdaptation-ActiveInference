function [C4disp] = Cfordisplay(Nm,qc)
%CFORDISPLAY  Policy laid back onto the monitor grid for display.
%   C4DISP = CFORDISPLAY(NM,QC) averages each cell's policy over the four
%   neighbours its actions lead to, so the 4 x Nm^4 policy QC can be drawn on
%   the Nm x Nm grid by FIGURE_OUTPUT_MAIN.  Edge and corner cells use their
%   reduced neighbourhoods: an edge cell averages over three neighbours and a
%   corner cell over two, the missing directions being replaced by the cell
%   itself.
%
%   The state here packs the hand and the target together (Nm^4 columns), so
%   the neighbour shifts are multiples of Nm^2.  CFORDISPLAY_VALIDATION is the
%   same routine for the hand-only state of the validation experiment.
%
%   See also CFORDISPLAY_VALIDATION, FIGURE_OUTPUT_MAIN.

% Calculations are performed separately for the cases of the corners, edges, and the inside.

% Corners
zeromatrix = zeros(1,Nm^2);
tm = zeromatrix;
tm(Nm+1:Nm:Nm*(Nm-2)+1)=1;
topmost = kron(tm, ones(1,Nm^2));
bm = zeromatrix;
bm(2*Nm:Nm:Nm*(Nm-1))=1;
bottommost = kron(bm,ones(1,Nm^2));
lm = zeromatrix;
lm(2:1:Nm-1)=1;
leftmost = kron(lm,ones(1,Nm^2));
rm = zeromatrix;
rm(Nm^2-Nm+2:1:Nm^2-1)=1;
rightmost = kron(rm,ones(1,Nm^2));

% Edges
tl = zeromatrix;
tl(1) = 1;
topleft =kron(tl,ones(1,Nm^2));
tr = zeromatrix;
tr(Nm^2-Nm+1) = 1;
topright=kron(tr,ones(1,Nm^2));
bl = zeromatrix;
bl(Nm) = 1;
bottomleft=kron(bl,ones(1,Nm^2));
br = zeromatrix;
br(Nm^2) = 1;
bottomright=kron(br, ones(1,Nm^2));

% Inside
is = ones(1,Nm^2);
is = is-tm-bm-lm-rm-tl-tr-bl-br;
inside = kron(is, ones(1,Nm^2));

insideC=(circshift(qc,[0,Nm^2])+circshift(qc,[0,Nm^3])+circshift(qc,[0,Nm^4-Nm^2])+circshift(qc,[0,Nm^4-Nm^3]))*0.25;
insideC = insideC.*inside;

% Combine contributions from all regions to form the final policy display
topmostC = (circshift(qc,[0,Nm^2])+circshift(qc,[0,Nm^3])+qc+circshift(qc,[0,Nm^4-Nm^3]))*0.25;
topmostC = topmostC .*topmost;
bottommostC = (qc+circshift(qc,[0,Nm^3])+circshift(qc,[0,Nm^4-Nm^2])+circshift(qc,[0,Nm^4-Nm^3]))*0.25;
bottommostC = bottommostC .*bottommost;
leftmostC =(circshift(qc,[0,Nm^2])+circshift(qc,[0,Nm^3])+circshift(qc,[0,Nm^4-Nm^2])+qc)*0.25;
leftmostC = leftmostC .*leftmost;
rightmostC =(circshift(qc,[0,Nm^2])+qc+circshift(qc,[0,Nm^4-Nm^2])+circshift(qc,[0,Nm^4-Nm^3]))*0.25;
rightmostC = rightmostC .*rightmost;

topleftC = 0.5 * qc + 0.25 *( circshift(qc,[0,Nm^2]) + circshift(qc,[0,Nm^3]) );
topleftC = topleft .* topleftC;
toprightC =0.5 * qc + 0.25 *( circshift(qc,[0,Nm^2]) + circshift(qc,[0,Nm^4 - Nm^3]) );
toprightC = topright .* toprightC;
bottomleftC = 0.5 * qc + 0.25 *( circshift(qc,[0,Nm^4 - Nm^2]) + circshift(qc,[0,Nm^3]) );
bottomleftC = bottomleft .* bottomleftC;
bottomrightC = 0.5 * qc + 0.25 *( circshift(qc,[0,Nm^4 - Nm^2]) + circshift(qc,[0,Nm^4 - Nm^3]) );
bottomrightC = bottomright .* bottomrightC;

C4disp = insideC + topmostC + bottommostC + leftmostC + rightmostC + topleftC + toprightC + bottomleftC + bottomrightC;
end
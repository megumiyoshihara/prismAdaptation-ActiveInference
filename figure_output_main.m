function figure_output_main(Nm, s, tX, tY, c)
%FIGURE_OUTPUT_MAIN  Draw one frame of the main experiment.
%   FIGURE_OUTPUT_MAIN(NM,S,TX,TY,C) draws the policy C as a colour per cell,
%   the target in white and the hand in black.  This is what MAIN_LEARNER
%   captures when cfg.record is set.
%
%   Each cell mixes the four action colours (up red, down cyan, left violet,
%   right yellow-green) in proportion to the policy there, so a cell whose
%   policy is certain comes out saturated and an undecided one comes out pale.
%   Saturation is boosted by 1.3 to keep the early, near-uniform policy visible.
%
%   The state S packs the hand and the target together, and C has one column per
%   packed state, so the target (TX,TY) selects the Nm^2 columns to display.
%
%   FIGURE_OUTPUT_MAIN(NM,S) leaves the policy out and shows the hand alone,
%   with the target overlaid at half intensity.  Mixing the colours costs far
%   more than the rest of the drawing, so this is the form to use where the
%   policy is not the point, as in the pretraining; (TX,TY) may be passed or
%   omitted, the target being decoded out of S either way.
%
%   See also FIGURE_OUTPUT_VALIDATION, CFORDISPLAY, MAIN_LEARNER.

rspS = reshape(s, Nm, Nm, Nm, Nm);
hand = reshape(sum(rspS, [1 2]),Nm,Nm);
[row,column] = find(hand == 1);
target = reshape(rspS(:, :, row,column), Nm, Nm);

if nargin < 5 || isempty(c)
    display = hand + 0.5*target;
    image(display,'CDataMapping','scaled');
    axis square;
    return;
end

hand = uint8(round(hand));
target = uint8(round(target));
targetXY = (tX-1) *Nm+tY;
colorUp = hsv2rgb([0,1,1]);
colorRight = hsv2rgb([0.25,1,1]);
colorLeft = hsv2rgb([0.75,1,1]);
colorDown = hsv2rgb([0.5,1,1]);
colors = [colorUp;colorDown;colorLeft;colorRight];
normC = param_normalization(c(:, targetXY:Nm^2:end), "A");
matrixC = reshape(normC' * colors, Nm, Nm, 3);
hsvmatrixC = rgb2hsv(matrixC);
% change Saturation
hsvmatrixC(:,:,2)= min(hsvmatrixC(:,:,2)*1.3,1) ;
% change Value
%hsvmatrixC(:,:,3) = 1;

rgbmatrixC = hsv2rgb(hsvmatrixC)*255;

RGB_C = uint8(round(rgbmatrixC));
hand(:,:,2) = hand;
hand(:,:,3) = hand(:,:,1);

target(:,:,2) = target;
target(:,:,3) = target(:,:,1);

display = RGB_C;
display(target==1) = 255;
display(hand==1) = 0;
image(display);
axis square;
txt = '{\color[rgb]{1, 0, 0}■\color[rgb]{0,0,0}up \color[rgb]{0,1,1}■\color[rgb]{0,0,0}down \color[rgb]{0.5,0,1}■\color[rgb]{0,0,0}left \color[rgb]{0.5,1,0}■\color[rgb]{0,0,0}right}';
subtitle(txt);
fontsize(gca,30,"pixels");
end

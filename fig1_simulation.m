%FIG1_SIMULATION  Minimal two-state model behind Figure 1, and its analytical curves.
%   Strips the task down to two states and two actions and runs Ntrial = 10
%   independent learners for T = 10000 trials, to show how fast the goal prior C
%   approaches the optimum Copt and that the rate matches the analytical
%   prediction.  Nothing here loads a simulation result; the whole figure is
%   generated from scratch.
%
%   Two update rules are compared, both driven by the same random trials:
%     C  (err1)  additive counts floored at 0.1, then normalised -- the rule the
%                naive learner in the full simulation uses.  Decays as 1/t^2.
%     C2 (err2)  additive counts pushed through a softmax.  Decays as exp(-t).
%   err1 and err2 are the squared distances to Copt, and th1 = 1/t^2 and
%   th2 = exp(-t) are the analytical curves drawn over them in red and blue.
%
%   Run as a script from the repository root.  It writes fig1_simulation.png
%   (log-log on top, semi-log below) and fig1_simulation.xlsx.  This is the only
%   script that produces the data Figure 1 is drawn from: paper_figures.py fig1()
%   reads the .xlsx out of PA_excel/.  The .png is the quick look, the .xlsx is
%   what the Python figure is plotted from.
%
%   See also PAPER_FIGURES (Python), NAIVE_LEARNING.

T = 10000;
Ns = 2;
Nd = 2;
Ntrial = 10;
c = zeros(Ns*Nd,T+1);
C = zeros(Ns*Nd,T);
Copt = [1,0,0,1]';
c2 = zeros(Ns*Nd,T);
C2 = zeros(Ns*Nd,T);
err1 = zeros(Ntrial,T);
err2 = zeros(Ntrial,T);
th1 = 1./(1:T).^2;
th2 = exp(-2*0.5*(0:T-1));

for i = 1:Ntrial
 qs = mnrnd(1,ones(Nd,1)/Nd,T)';
 qd = mnrnd(1,ones(Nd,1)/Nd,T)';
 Gamma = 1-sum(qs.*qd);
 for t = 1:T
  c(:,t+1) = max(c(:,t) + (1-2*Gamma(t))*kron(qs(:,t),qd(:,t)), 0.1);
  C(:,t) = c(:,t+1) ./ ([eye(Nd),eye(Nd);eye(Nd),eye(Nd)]*c(:,t+1));
  c2(:,t+1) = c2(:,t) + (1-2*Gamma(t))*kron(qs(:,t),qd(:,t));
  C2([1,3],t) = exp(c2([1,3],t)-max(c2([1,3],t)))/sum(exp(c2([1,3],t)-max(c2([1,3],t))));
  C2([2,4],t) = exp(c2([2,4],t)-max(c2([2,4],t)))/sum(exp(c2([2,4],t)-max(c2([2,4],t))));
 end
 err1(i,:) = sum((C-Copt).^2);
 err2(i,:) = sum((C2-Copt).^2);
end

fig               = figure();
fig.Position(3:4) = [300 600];
subplot(2,1,1), loglog(1:T,err1','k-',1:T,err2','k-'), hold on, loglog(1:T,th1,'r-',1:T,th2,'b-','LineWidth',4), hold off, axis([0 T 10^-10 1])
subplot(2,1,2), semilogy(1:T,err1','k-',1:T,err2','k-'), hold on, semilogy(1:T,th1,'r-',1:T,th2,'b-','LineWidth',4), hold off, axis([0 100 10^-10 1])
print(fig, "fig1_simulation.png", '-dpng')

%%{
xls_filename = "fig1_simulation.xlsx";
writematrix(err1, xls_filename, 'Sheet','err1', 'Range','B2');
writematrix(1:T,  xls_filename, 'Sheet','err1', 'Range','B1');
writematrix((1:10)', xls_filename, 'Sheet','err1', 'Range','A2');
writematrix(err2, xls_filename, 'Sheet','err2', 'Range','B2');
writematrix(1:T,  xls_filename, 'Sheet','err2', 'Range','B1');
writematrix((1:10)', xls_filename, 'Sheet','err2', 'Range','A2');
writematrix(th1, xls_filename, 'Sheet','th1', 'Range','B2');
writematrix(1:T, xls_filename, 'Sheet','th1', 'Range','B1');
writematrix(1,   xls_filename, 'Sheet','th1', 'Range','A2');
writematrix(th2, xls_filename, 'Sheet','th2', 'Range','B2');
writematrix(1:T, xls_filename, 'Sheet','th2', 'Range','B1');
writematrix(1,   xls_filename, 'Sheet','th2', 'Range','A2');
%}
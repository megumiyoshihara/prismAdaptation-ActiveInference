function [varargout] = param_normalization(X, norm_type)
%PARAM_NORMALIZATION  Normalise a Dirichlet parameter matrix.
%   [X_,LNX] = PARAM_NORMALIZATION(X,NORM_TYPE) divides X by its column sums
%   (NORM_TYPE "A") or its row sums (NORM_TYPE "B"), flooring the denominator at
%   1e-6.  The second output is psi(X) - psi(sum X), the expected log of the
%   Dirichlet, which is what the learners accumulate in log space.
%
%   Only "A" and "B" are implemented; any other NORM_TYPE leaves sumOfX unset.
varargout{1} = 0;
if norm_type=="A"
    sumOfX = sum(X);

elseif norm_type=="B"
    sumOfX = sum(X, 2);
end
varargout{1}=X./max(10^-6,sumOfX);
varargout{2}= psi(X)-psi(sumOfX);

end
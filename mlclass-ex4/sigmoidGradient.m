function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).





% g'(z) = d/z of g(z) = g(z)(1 - g(z))
% sigmoid(z) = g(z) = 1 / (1 + e^-z)

%g = max(0, z);
sig = sigmoid(z);
g = sig .* (1- sig);








% =============================================================




end

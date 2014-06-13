function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %



    %total = 0;
    %for i=1:m,
    %  x_sub_i = X(i,:)';
    %  real_num = (theta' * x_sub_i) - y(i,:);
    %  total += (real_num) * x_sub_i;
    %end
    %total = total / (m);
    %alphaResult = alpha * total;
    total = (((X * theta) - y)' * X)' / m;
    alphaResult = alpha * total;
    theta = theta - alphaResult;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end

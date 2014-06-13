function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ones(m,1) X];
layer2_activation = [ones(m,1) sigmoid(X * Theta1')];
%{
layer2Output = zeros(m, hidden_layer_size);
sigmoid(X * Theta1(j,:)');
a2 = zeros(m, hidden_layer_size);
for j=1:hidden_layer_size
    layer2Output(:,j) = sigmoid(X * Theta1(j,:)');
endfor
layer2Output = [ones(m,1) layer2Output];
%}


%J += ((log(a3)' * Y) + (log(1 - a3)' * (1-Y))) / -m;
layer3_activation = sigmoid(layer2_activation * Theta2');
Y = (eye(num_labels)(y,:));
for k=1:num_labels
    yk = Y(:,k);
    %a3 = sigmoid(layer2_activation * Theta2(k,:)');
    layer3_activation_k = layer3_activation(:,k);
    J += ((log(layer3_activation_k)' * yk) + (log(1 - layer3_activation_k)' * (1-yk))) / -m;
endfor

%{
for i=1:m
    for k=1:num_labels
        yk = y == k;
        thetak = Theta2(k,:);
	sig = sigmoid(thetak * layer2Output(i,:)');
	yki = yk(i);
        J += ((yki * log(sig)) + ((1-yki) * log(1-sig))) / -m;
    endfor
endfor
%}

J += (lambda / (2 * m)) * (nn_params' * nn_params); % in this scenarios we are regularizing the bias units
%costtheta1 = sum(Theta1(:,2:size(Theta1,2))(:) .^ 2);
%costtheta2 = sum(Theta2(:,2:size(Theta2,2))(:) .^ 2);
%J += (lambda / (2 * m)) * (costtheta1 + costtheta2);

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

for t=1:m
    a_1 = X(t,:)'; % input in layer 1 + 1 bias unit
    			% each row in this vector cooresponds to a unit in layer 1

			% input values + 1 bias unit
			% where each input value row cooresponds to a single unit
			% in this layer 1 (or input layer)

    z_2 = Theta1 * a_1; % weighted input mapping from layer 1 to layer 2
			% each row = 
			% sum of weighted values provided from every unit 
			% in layer 1)

    a_2 = [1; sigmoid(z_2)]; % activation of layer 2 input + 1 bias unit
		   	     % each row = 
			     % activation of (sum of weighted values provided from 
			     % every unit in layer 1) + 1 bias unit

    z_3 = Theta2 * a_2; % weighted input mapping from layer 2 to layer 3
			% each row = 
			% sum of weighted values provided from every activated unit in 
			% layer 2

    a_3 = sigmoid(z_3); % acivation of layer 3 input
			% each row = 
			% activation of (sum of weighted values provided from 
			% every unit in layer 2)

    yk = zeros(num_labels,1);
    yk(y(t,1),1) = 1; % binary vector for whether training example belongs to class k
    d_3 = a_3 - yk; % delta of proposed predicted activation in layer 3 vs actual yk


    e_2 = Theta2' * d_3; % sum of weighted delta error (prediction - actual) for each unit in layer 2
    			 % each row = 
			 % sum of weighted errors for every weight applied from a single unit in layer 2 to 
			 % units in layer 3
    d_2 = e_2 .* [1; sigmoidGradient(z_2)];
    			 % using the sum of weighted values provided from every unit in layer 1
			 % we want to to take the derivative of the activation of those values
			 % such that we are determining where along the activation in terms of 
			 % gradient, each of those weighted values sit,
			 % then multiple each gradient by the error

    d_2 = d_2(2:end); % remove the first delta since this cooresponds to our extra bias gradient we introduced above

    Theta2_grad += d_3 * (a_2)';
    Theta1_grad += d_2 * (a_1)';
endfor

Theta2_grad /= m;
Theta1_grad /= m;

adjust = lambda / m;
Theta2_grad += [zeros(num_labels,1) Theta2(:,2:size(Theta2,2))] * adjust;
Theta1_grad += [zeros(hidden_layer_size,1) Theta1(:,2:size(Theta1,2))] * adjust;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

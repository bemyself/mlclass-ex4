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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(size(X,1), 1) X];
z2 = Theta1*X';
a2 = sigmoid(z2);
a2 = [ones(1, size(a2,2)); a2];
z3 = Theta2*a2;
a3 = sigmoid(z3);

y_index = zeros(num_labels, m);
for i = 1:m
    y_index(y(i), i) = 1;

    J = J + sum(-y_index(:,i).*log(a3(:,i))-(1-y_index(:,i)).*log(1-a3(:,i)));
end

J = J/m;

[m1, n1] = size(Theta1);
[m2, n2] = size(Theta2);

N_Theta1 = Theta1(:, 2:n1);
N_Theta2 = Theta2(:, 2:n2);

squ_sum1 = 0;
squ_sum2 = 0;

for i = 1:m1
    for j = 1:(n1-1)
        squ_sum1 = squ_sum1 + N_Theta1(i,j)^2;
    end
end

for i = 1:m2
    for j = 1:(n2-1)
        squ_sum2 = squ_sum2 + N_Theta2(i,j)^2;
    end
end

J = J + (squ_sum1 + squ_sum2)*lambda/(2*m);
a1 = X;

for t = 1 : m
    delta3(:, t) = a3(:, t) - y_index(:, t);
    g_deri(:,t) = [1; sigmoidGradient(z2(:,t))];
    delta2(:, t) = Theta2'*delta3(:,t).*g_deri(:,t);
    N_delta2(:, t) = delta2(2:end, t);
    Theta2_grad = Theta2_grad + delta3(:,t)*a2(:,t)';
    Theta1_grad = Theta1_grad + N_delta2(:,t)*a1(t,:); 
end

Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

Temp1 = Theta1;
Temp2 = Theta2;

Temp1(:,1) = 0;
Temp2(:,1) = 0;

Theta1_grad = Theta1_grad + Temp1*lambda/m;
Theta2_grad = Theta2_grad + Temp2*lambda/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

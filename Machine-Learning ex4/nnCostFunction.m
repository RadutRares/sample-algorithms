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

eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

a1 = [ones(size(X),1), X];

z2 = a1 * Theta1';

a2 = sigmoid(z2);

a2 = [ones(size(a2),1), a2];

z3 =  a2 * Theta2';

% Forwards Propogation complete
a3 = sigmoid(z3);

h = a3;

% For the classification section
% Have to perform element wise multiplication due to the nature of the y_matrix
% Thus afterwards you must compute the double sum
% For regularization must remove the first column of each theta to eliminate the bias columns
% Afterwards, roll into a vector to sum

J = sum(sum( -y_matrix .* log(h) - (1-y_matrix) .* log(1 - h))) / m  + (lambda / 2)*(sum( [Theta1(:,2:end)(:); Theta2(:,2:end)(:)] .^ 2)) / m;


% Begin Backpropagation

d3 = a3 - y_matrix;

d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);

Delta1 = d2' * a1;

Delta2 = d3' * a2; 

Theta1_grad = Delta1 / m;
Theta2_grad = Delta2 / m;

% Regularization, use old Theta
Theta1(:,1) = 0;
Theta2(:,1) = 0;

% Add lambda value for all of Theta
Theta1_grad = Theta1_grad + Theta1 * (lambda / m);
Theta2_grad = Theta2_grad + Theta2 * (lambda / m);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

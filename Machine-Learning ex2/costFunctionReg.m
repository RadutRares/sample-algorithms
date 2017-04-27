function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of features

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Vectorized Cost Function
h = sigmoid(X * theta);
J = ( -y' * log(h) - (1-y)' * log(1 - h)) / m;

J_Lambda = 0;
for j = 2:n
  J_Lambda = J_Lambda + theta(j)^2;
end
J_Lambda = (lambda / (2 * m)) * J_Lambda;

J = J + J_Lambda;

% Vectorized Gradients
g = sigmoid(X * theta);
grad = (X' * (g - y))/ m;

for j = 2:n
  grad(j) = grad(j) + (lambda / m) * theta(j);
end

% =============================================================

end

function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values

m = length(y); % number of training examples

% Vectorized Cost Function
h = sigmoid(X * theta);
J = ( -y' * log(h) - (1-y)' * log(1 - h)) / m;

% Vectorized Gradients
g = sigmoid(X * theta);
grad = (X' * (g - y))/ m;

% Not Vectorized Implementation
%J2 = 0;
%for i=1:m 
%  x = X(i,:);
%  h = sigmoid(x * theta);
%  J2 = J2 + ( -y' * log(h) - (1-y)' * log(1 - h));
%end
%J2 = J2 ./ m;
%J2 = sum(J2)

end

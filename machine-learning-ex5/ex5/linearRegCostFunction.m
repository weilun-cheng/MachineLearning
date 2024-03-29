function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta,1);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
J = 0.5/m*sum((X*theta-y).^2)+0.5*lambda/m*sum(theta(2:n,:).^2);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
prediction = X*theta;
temp = X(:,1)' * (prediction-y)/m;
grad=X'*(prediction-y)/m+lambda/m*theta;
grad(1,:)=temp;









% =========================================================================

grad = grad(:);

end

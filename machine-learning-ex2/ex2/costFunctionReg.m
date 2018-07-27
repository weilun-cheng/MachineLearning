function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
prediction=sigmoid(X * theta);
delta = y.*log(prediction)+(1-y).*log(1-prediction);
J=-1/m*sum(delta)+lambda*0.5/m*sum(theta(2:n,:).^2);
grad(1,:) = X(1:m,1)'*(prediction-y)/m;
for i=2:n
    grad(i,:) = X(1:m,i)' * (prediction-y)/m + lambda/m*theta(i);
end
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end

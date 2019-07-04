function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
j = 1:m;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


    t1 = sum((theta(1) + (theta(2) .* X(j,2)) + (theta(3) .* X(j,3))) - y(j))  % (1X1)+(1X1).*(1X1) --->  (97X1)-(97X1) ---> sum(97X1) ---> (1X1)
    t2 = sum(((theta(1) + (theta(2) .* X(j,2)) + (theta(3) .* X(j,3)) - y(j)) .* X(j,2)))  % (1X1)+(1X1).*(1X1) --->  (97X1)-(97X1).*(97X1) ---> sum(97X1) ---> (1X1)
    t3 = sum(((theta(1) + (theta(2) .* X(j,2)) + (theta(3) .* X(j,3)) - y(j)) .* X(j,3)))  % (1X1)+(1X1).*(1X1) --->  (97X1)-(97X1).*(97X1) ---> sum(97X1) ---> (1X1)
    
    theta(1) = theta(1) - (alpha/m) * t1;
    theta(2) = theta(2) - (alpha/m) * t2;
    theta(3) = theta(3) - (alpha/m) * t3;
    


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end

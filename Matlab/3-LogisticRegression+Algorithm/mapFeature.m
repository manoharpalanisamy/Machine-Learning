function out = mapFeature(X1, X2)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%


% CLEAR By Workout
degree = 6;
out = ones(size(X1(:,1))); % set Ones to 1st column 
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);  % end value is 1 already contains column vector 1 , end+1 --> (1+1) from 2 second column assign upto 27 column
    end
end

end
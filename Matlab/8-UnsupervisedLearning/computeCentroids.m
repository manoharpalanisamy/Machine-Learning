function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X); % (300x2)

% You need to return the following variables correctly.
centroids = zeros(K, n); % (3,2)


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%


for k=1:K % for-loop over the centroids 
   centroids(k, :) = mean(X(idx==k, :)); % idx index of closest centroids is computed centroid from findClosestCentroids.m 
                                         % Go over every centroid idx and compute mean of all points that belong to it
                                         % mean(X(idx==k, :))
                                         % 1) idx==k (300x1)==1 --> it
                                         % returns value if idx==1 or 2 or
                                         % 3
                                 
                                      
end

% =============================================================


end


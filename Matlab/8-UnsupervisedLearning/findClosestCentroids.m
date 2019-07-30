function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1); % set cluster K=3 

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);  %  CLEAR (300x1) we have to set each training eg to 300 centroids  

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m = size(X,1); % 300

for i = 1:m
    distance_array = zeros(1,K); % 3
%     fprintf("\ni=[%d]\t",i)
    for j = 1:K
       % fprintf("j=[%d]",j)
        distance_array(1,j) = sqrt(sum(power((X(i,:)-centroids(j,:)),2)));  % Centroids distance from each X(i)  , square we uses power((x-u),2),
        % summation uses sum() , sqrt no idea  
        % each training eg we get 3 centroids because we have K=3 clusters 
    end
    [~, d_idx] = min(distance_array); % [temp1 , temp_idx] = min(temp)   temp1 = 1.9812 temp_idx = 1 , temp1 contains min value , 
    %  temp_idx returns index of min value ,
    %  [~, d_idx]  ~ is used for ignorance variable , d_idx contains index
    %  of min value
    % https://in.mathworks.com/matlabcentral/answers/73735-what-does-it-mean-by-writing-idx-in-code
    
    idx(i,1) = d_idx;
end






% =============================================================

end


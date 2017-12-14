function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
m = size(X,1);

idx = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%


dist = zeros(m, K);
for i = 1:K
  column = bsxfun(@minus, X, centroids(i,:)); % All features subtracted by centroid
  dist(:,i) = sum(column.^2,2); %Summed into columns
end

for i = 1:m
  [v,id] = min(dist(i,:)); % Smallest index of each row is correct centroid
  idx(i) = id;
end

SideBySide = idx;

dist = zeros(m, K);

%
for i = 1:m
  for j = 1:K
    dist(i,j) = sum((X(i,:) - centroids(j,:)).^2);
  end
  [v,id] = min(dist(i,:));
  idx(i) = id;
end

SideBySide = [SideBySide, idx];
SideBySide(4:6,:)


#{
for i = 1:K
  column = bsxfun(@minus, X, centroids(i,:)) % All features subtracted by centroid
  dist(:,i) = sum(column.^2,2); %Summed into columns
end

for i = 1:K
  [v,id] = min(dist(i,:)); % Smallest index of each row is correct centroid
  idx(i) = id;
end
}#

% =============================================================

end


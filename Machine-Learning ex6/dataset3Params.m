function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%


% Original Lists
% C_Vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% sigmaVec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

C_Vec = [0.9, 1, 1.1];
sigmaVec = [0.09, 0.1, 0.11];
predictions = 0;


lowest_error = 1;
bestC = 0;
bestSig = 0;

for C = C_Vec
  for sigma = sigmaVec
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    if error < lowest_error
      bestC = C;
      bestSig = sigma;
      lowest_error = error;
    end
   end 
end

C = bestC;
sigma = bestSig;
% =========================================================================

end

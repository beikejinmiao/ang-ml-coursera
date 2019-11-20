function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Cs = [0.1, 0.4, 1, 8, 16, 64, 128, 512, 1024];
sigmas = [0.02, 0.1, 0.2, 0.4, 1, 1.6, 6.4, 12.8, 51.2];
errors = zeros(length(Cs), length(sigmas));
for i=1:length(Cs)
  C = Cs(i);
  for j=1:length(sigmas)
    sigma = sigmas(j);
    [model] = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    preds = svmPredict(model, Xval);
    errors(i,j) = mean(double(preds ~= yval));
  endfor
endfor

min_error = min(min(errors));
[i, j] = find(min_error == errors);
C = Cs(i);
sigma = sigmas(j);

# best
# C =  1, sigma =  0.10000 for error=0.03

% =========================================================================

end

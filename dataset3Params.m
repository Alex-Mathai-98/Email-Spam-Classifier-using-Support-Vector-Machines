function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. 

 
%The code to check it out
#{
error_value = exp(100000);
error_min_parameters = [0,0];
c = 0.01;
while c <= 1000,
  Sigma = 0.01;
  while Sigma <= 1000,
    [model] = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, Sigma));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    fprintf('The current error is %f', error);
    if error < error_value,
      error_value = error;
      error_min_parameters(1,1) = c;
      error_min_parameters(1,2) = Sigma;
    end;
    fprintf('The values of C and sigma are %f %f', c,Sigma);
    Sigma = Sigma*3;
  end;
  c = c*3;
end;

C = error_min_parameters(1,1);
sigma = error_min_parameters(1,2);
#}
C =1;
sigma = 0.1;
% =========================================================================

end

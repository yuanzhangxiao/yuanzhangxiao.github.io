% Load the data from the MNIST
% We only load the data containing digits 5 and 6
% Input arguments:
%   number_training: number of training examples

function [train_image, train_label, test_image, test_label] = load_mnist_5_6(number_training)

  % Set the seed of the random generator
  rng(0, 'twister');

  % Load the training data
  A=loadMNISTImages('train-images-idx3-ubyte');
  b=loadMNISTLabels('train-labels-idx1-ubyte')';

  % Take only the digits 5 and 6
  A = [ A(:,b==5), A(:,b==6) ];
  b = [ b(b==5), b(b==6) ];
  
  % Set the 5,6 digits to be 0,1 (for the logistic regression)
  b(b==5) = 0;
  b(b==6) = 1;

  % Randomly shuffle the data
  I = randperm(length(b));
  b = b(I); % labels in range 1 to 10
  A = A(:,I);
  
  % Take the first (number_training) training examples
  A = A(:, 1:number_training);
  b = b(1:number_training);

  % We standardize the data so that each pixel will have roughly zero mean and unit variance.
  s=std(A,[],2);
  m=mean(A,2);
  A=bsxfun(@minus, A, m);
  A=bsxfun(@rdivide, A, s+.1);

  % Place these in the training set
  train_image = A;
  train_label = b;

  % Load the testing data
  A=loadMNISTImages('t10k-images-idx3-ubyte');
  b=loadMNISTLabels('t10k-labels-idx1-ubyte')';

  % Take only the digits 5 and 6
  A = [ A(:,b==5), A(:,b==6) ];
  b = [ b(b==5), b(b==6) ];
  
  % Set the 5,6 digits to be 0,1 (for the logistic regression)
  b(b==5) = 0;
  b(b==6) = 1;

  % Randomly shuffle the data
  I = randperm(length(b));
  b = b(I); % labels in range 1 to 10
  A = A(:,I);

  % Standardize using the same mean and scale as the training data.
  A = bsxfun(@minus, A, m);
  A = bsxfun(@rdivide, A, s+.1);

  % Place these in the testing set
  test_image = A;
  test_label = b;
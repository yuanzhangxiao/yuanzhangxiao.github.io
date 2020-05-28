% Load the data from the MNIST
% Input arguments:
%   binary_digits: true or false, if true, load the digits of 0 and 1 only
%   number_training: number of training examples

function [train_image, train_label, test_image, test_label] = load_mnist(binary_digits, number_training)

  % Set the seed of the random generator
  rng(0, 'twister');

  % Load the training data
  A=loadMNISTImages('train-images-idx3-ubyte');
  b=loadMNISTLabels('train-labels-idx1-ubyte')';

  if (binary_digits)
    % Take only the 0 and 1 digits
    A = [ A(:,b==0), A(:,b==1) ];
    b = [ b(b==0), b(b==1) ];
  end

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

  if (binary_digits)
    % Take only the 0 and 1 digits
    A = [ A(:,b==0), A(:,b==1) ];
    b = [ b(b==0), b(b==1) ];
  end

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
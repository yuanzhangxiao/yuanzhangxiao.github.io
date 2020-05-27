% Load the MNIST data for this exercise.
% train_image and test_image will contain the training and testing images.
% train_image has size [n,m] where:
%     n is the number of pixels in each image.
%     m is the number of examples.
% train_label and test_label will contain the corresponding labels (0 or 1).

% Use the next two lines for detecting digits 0 and 1
binary_digits = true;
[train_image, train_label, test_image, test_label] = load_mnist(binary_digits, 10);

% Use the next line for detecting digits 5 and 6
% [train_image, train_label, test_image, test_label] = load_mnist_5_6(50);

% Add row of 1s to the dataset to act as an intercept term
train_image = [ones(1,size(train_image,2)); train_image]; 
test_image = [ones(1,size(test_image,2)); test_image];

% Training set dimensions
m=size(train_image,2);
n=size(train_image,1);

% Initial value of the coefficients
x = zeros(n,1);

% ======================================================================
% Computer the coefficients x in the logistic regressor!


% ======================================================================

% Print out the accuracy of your trained logistic regressor.
accuracy =sum( test_label == ( 1./( 1 + exp( - x' * test_image ) ) > 0.5) ) / length(test_label);
fprintf('Training accuracy: %2.1f%%\n', 100*accuracy);
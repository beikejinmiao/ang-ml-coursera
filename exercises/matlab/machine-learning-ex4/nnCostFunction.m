function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

# J第一种计算方法
%{
yeye = eye(num_labels);             
X = [ones(m,1) X];         # add bias unit
for i=1:m
    z2 = X(i, :) * Theta1';         # hidden layer
    a2 = sigmoid(z2);
    a2 = [ones(size(a2,1), 1) a2];  # add bias unit
    z3 = a2 * Theta2';              # output layer
    a3 = sigmoid(z3);
    hi = a3;                        # hypothesis 
    yi = yeye(y(i), :);             # true value
    for k=1:num_labels
        J -= yi(k)*log(hi(k)) + (1-yi(k))*log(1-hi(k));
    endfor
endfor
J /= m;
%}

# J第二种计算方法
z2 = [ones(m,1) X] * Theta1';
a2 = sigmoid(z2);      
z3 = [ones(m, 1) a2] * Theta2';          
a3 = preds = sigmoid(z3);      
# size(Theta1) = (25, 401)    # size(Theta2) = (10, 26)
# size(a2) = (5000,25)        # size(a3) = (5000,10)
%{
for i=1:m
  yi = zeros(1, num_labels);  # 将yi转成行向量
  yi(y(i)) = 1;
  hi = preds(i, :)';
  J -= yi*log(hi) + (1-yi)*log(1-hi);
endfor
J /= m;
%}

# J第三种计算方法
y_mat = zeros(m, num_labels);
for i=1:m
  y_mat(i, y(i)) = 1;       # 将y转成矩阵
endfor
J = -trace((y_mat*log(preds') + (1-y_mat)*log(1-preds'))) / m;



# 正则化第一种计算方法
%{
# 计算正则化
reg = 0;
for j=1:25
  for k=1:400
    reg += Theta1(j, k)^2;
  endfor
endfor
for j=1:10
  for k=1:25
    reg += Theta2(j, k)^2;
  endfor
endfor
reg = lambda * reg /2 / m
%}

# 正则化第二种计算方法
## 正则化不包含偏置项的参数 
reg = lambda * (sum(power(Theta1(:, 2:end), 2)(:)) + sum(power(Theta2(:, 2:end), 2)(:))) /2/m;    # regularized
J = J + reg;


# https://github.com/atinesh-s/Coursera-Machine-Learning-Stanford/blob/master/Week%205/Programming%20Assignment/machine-learning-ex4/ex4/nnCostFunction.m
%{
for i=1:m
  # 第一步
  yi = zeros(num_labels,1);  # 将yi转成列向量
  yi(y(i)) = 1;
  a1 = [1 X(i, :)];   # add a bias (1*401)
  a1 = a1';           # (1*401) -> (401*1)
  z2 = Theta1 * a1;   # (25*401) * (401*1) = （25*1）
  a2 = sigmoid(z2);   # (25*1)
  a2 = [1; a2];       # add a bias (26*1)
  z3 = Theta2 * a2;   # (10*26) * (26*1)
  a3 = sigmoid(z3);   # final activation layer a3 == h(theta) (10*1)
  hi = a3;
  # 第二步
  delta3 = (hi-yi);   # (10*1)
  z2=[1; z2];         # add a bias (26*1)
  # 第三步
  delta2 = (Theta2' * delta3) .* sigmoidGradient(z2);  # (26*20) * (10*1) = =(26*1)
  # 第四步
  delta2 = delta2(2:end);   # (26*1) -> (25*1) 
  Theta2_grad += delta3 * a2';    # (10*1) * (1*26) = (10*26) 
  Theta1_grad += delta2 * a1';    # (25*1) * (1*401) = (25*401)
endfor

Theta2_grad = (1/m) * Theta2_grad;
Theta1_grad = (1/m) * Theta1_grad;
Theta1_grad(:, 2:end) += (lambda/m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) += (lambda/m) * Theta2(:, 2:end); 
%}



# https://github.com/tuanavu/coursera-stanford/blob/master/machine_learning/exercises/matlab/machine-learning-ex4/ex4/nnCostFunction.m

X = [ones(m , 1)  X];

% Part 1: CostFunction
% -------------------------------------------------------------

a1 = X;
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m , 1)  a2];
a3 = sigmoid(a2*Theta2');

ry = eye(num_labels)(y,:);

cost = ry.*log(a3) + (1 - ry).*log(1 - a3);
J = -sum(sum(cost,2)) / m;

reg = sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(: , 2:end).^2));

J = J + lambda/(2*m)*reg;

% -------------------------------------------------------------

% Part 2: Backpropagation algorithm
% -------------------------------------------------------------


delta3 = a3 - ry;            # Q1：为什么预测值减去实际值就是
delta2 = (delta3*Theta2)(:,2:end) .* sigmoidGradient(z2);

Delta1 = delta2'*a1;
Delta2 = delta3'*a2;

Theta1_grad = Delta1 / m + lambda*[zeros(hidden_layer_size , 1) Theta1(:,2:end)] / m;
Theta2_grad = Delta2 / m + lambda*[zeros(num_labels , 1) Theta2(:,2:end)] / m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

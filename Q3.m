%% Cleaning Up
% Close all previously opened figueres and clear workspace
close all;
clear;


%% Load Data
load ./olympics.mat

x = male100(:,1);
t = male100(:,2);


%% Initialise constants

K = 5; %K-fold CV

% Generate vector of lambda values to check
lambda_max = 1000;
% Since 1st element is 0
second_element = 1e-10;
% Factor ratio between consecutive almbda elements
factor = 2;
num_lambdas = log(lambda_max/second_element)/log(factor);

% Create the lambda vector with elements
% ranging from 0 to num_lambdas
lambda_values = second_element * factor.^(0:num_lambdas);
lambda_values = [0 lambda_values];

%% Find split indices of X for cross-validation

N = length(t);
X = []; 
sizes = repmat(floor(N/K),1,K);
sizes(end) = sizes(end) + N - sum(sizes);
csizes = [0 cumsum(sizes)];

%% Linear Model

% Generate feature matrix X
for k = 0:1
    X = [X x.^k];
end

% Iterate over all lambda in lambda_values
for i = 1:length(lambda_values)
    lambda = lambda_values(i);
    for fold = 1:K
        % Partition the data
        
        % Data subset corresponding to test set
        foldX = X(csizes(fold)+1:csizes(fold+1),:);
        foldt = t(csizes(fold)+1:csizes(fold+1));
        % Data subset corresponding to training set
        trainX = X;
        trainX(csizes(fold)+1:csizes(fold+1),:) = [];
        traint = t;
        traint(csizes(fold)+1:csizes(fold+1)) = [];
        
        dim = size(trainX);
        n = dim(1);

        % Regularised Least Square solution is 
        % w = (X'X + N*lamda*I)^-1 * X' * t 
        % where (X' is X transpose) & (I is identity matrix)
        w = inv(trainX'*trainX + n*lambda*eye(2))*trainX'*traint;
        fold_pred = foldX*w;
        cv_loss_1(fold,i) = mean((fold_pred-foldt).^2);
    end
end

%% Find Best Lambds - Linear Model

% Find the value of lambda for which mean(cv_loss) is minimum
min_mean_cv_loss = min(mean(cv_loss_1, 1));
optimal_lambda_index = find(mean(cv_loss_1, 1) == min_mean_cv_loss);
optimal_lambda_1 = lambda_values(optimal_lambda_index);


%% Plotting Loss - Linear Model

figure(1);
plot(lambda_values, mean(cv_loss_1,1),'linewidth',2)
xlabel('Lambda','fontsize',15);
ylabel('Loss','fontsize',15);
set(gca, 'XScale', 'log');
title('CV Loss - Linear Model','fontsize',20);

%% 4th Order Model

% Generate feature matrix X
for k = 2:4
    X = [X x.^k];
end

% Iterate over all lambda in lambda_values
for i = 1:length(lambda_values)
    lambda = lambda_values(i);
    for fold = 1:K
        % Partition the data
        
        % Data subset corresponding to test set
        foldX = X(csizes(fold)+1:csizes(fold+1),:);
        foldt = t(csizes(fold)+1:csizes(fold+1));
        % Data subset corresponding to training set
        trainX = X;
        trainX(csizes(fold)+1:csizes(fold+1),:) = [];
        traint = t;
        traint(csizes(fold)+1:csizes(fold+1)) = [];
        
        dim = size(trainX);
        n = dim(1);

        % Regularised Least Square solution is 
        % w = (X'X + N*lamda*I)^-1 * X' * t 
        % where (X' is X transpose) & (I is identity matrix)
        w = inv(trainX'*trainX + n*lambda*eye(5))*trainX'*traint;
        fold_pred = foldX*w;
        cv_loss_4(fold,i) = mean((fold_pred-foldt).^2);
    end
end

%% Find Best Lambds - 4th Order Model

% Find the value of lambda for which mean(cv_loss) is minimum
min_mean_cv_loss = min(mean(cv_loss_4, 1));
optimal_lambda_index = find(mean(cv_loss_4, 1) == min_mean_cv_loss);
optimal_lambda_4 = lambda_values(optimal_lambda_index);

%% Plotting Loss - 4th Order Model

figure(2);
plot(lambda_values, mean(cv_loss_4,1),'linewidth',2)
xlabel('Lambda','fontsize',15);
ylabel('Loss','fontsize',15);
set(gca, 'XScale', 'log');
title('CV Loss - 4th Order Model','fontsize',20);


%% Display best lambda values

disp(['Optimal lambda for Linear Model: ', num2str(optimal_lambda_1)]);
disp(['Optimal lambda for 4th Order Model: ', num2str(optimal_lambda_4)]);
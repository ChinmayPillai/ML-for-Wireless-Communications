%% Cleaning Up
% Close all previously opened figueres and clear workspace
close all;
clear;


%% Part A - Calculating 't'
% Initialise coefficients, N and order
w0 = 0;
w1 = 1;
w2 = -1;
w3 = 5;
N = 100;
var = 300;
xMax= 5;

% Generate random noise column vector from standard normal distribution
n = sqrt(var) * randn(N, 1);

% Generate x from uniform distributrion from -xMax to +xMax
x = 2*xMax*(rand(N,1) - 0.5);
% Sort x
x = sort(x);

% Find t vector
t = w0 + w1*x + w2*(x.^2) + w3*(x.^3) + n;

figure(1);
% Plot generated data
plot(x,t,'k.','markersize', 8);
xlim([-xMax xMax])
xlabel('$x$','interpreter','latex','fontsize',15);
ylabel('$t$','interpreter','latex','fontsize',15);
hold on

% Model Prediction values
x_values = [-5:0.1:5]';

% Generate feature matrix for train and test data
% X_train and features respectively
% [1 x x^2 x^3 x^4 x^5 x^6]

X_train = [];
features = [];
for k = 0:6
    X_train = [X_train x.^k];
    features = [features x_values.^k];
end


%% Part B - Fitting models

% Model the data as
% t = w'*X + e
% where e is gaussian noise with 0 mean and some variance var

% MLE solution is 
% w_mle = (X'X)^-1 * X' * t (X' is X transpose)
% var_mle = (1/N) * (t't - t'X * w_mle)

% Model on Prediction
% t_mle = w'x_n = x_n'*w
% err_mle = var_mle * x_n' * (X'X)^-1 * x_n


%% Part B - Fitting linear model

X = X_train(:, 1:2);
x_pred = features(:, 1:2); 
w_linear = inv(X'*X) * X' * t;
var_mle_linear = (1/N)*(t'*t - t'*X*w_linear);

% Find predicted model values
t_linear = x_pred * w_linear;
err_linear = var_mle_linear * diag(x_pred*inv(X'*X)*x_pred');

%% Part B - Fitting Cubic model

X = X_train(:, 1:4);
x_pred = features(:, 1:4);
w_cubic = inv(X'*X) * X' * t;
var_mle_cubic = (1/N)*(t'*t - t'*X*w_cubic);

% Find predicted model values
t_cubic = x_pred * w_cubic;
err_cubic = var_mle_cubic * diag(x_pred*inv(X'*X)*x_pred');

%% Part B - Fitting 6th order model

X = X_train;
x_pred = features;
w_sixth = inv(X'*X) * X' * t;
var_mle_sixth = (1/N)*(t'*t - t'*X*w_sixth);

% Find predicted model values
t_sixth = x_pred * w_sixth;
err_sixth = var_mle_sixth * diag(x_pred*inv(X'*X)*x_pred');


%% Part C - Plot the models

% Create model prediction, error and titles vectors to interate through
pred = [t_linear t_cubic t_sixth];
err = [err_linear err_cubic err_sixth];
titles = {'Linear Model - Order 1', 'Cubic Model - Order 3', '6th Order Model'};

for i = 1:3

    % Plot the data and predictions
    figure(i);
    hold off
    plot(x,t,'k.','markersize',10);
    xlabel('$x$','interpreter','latex','fontsize',15);
    ylabel('$t$','interpreter','latex','fontsize',15);
    hold on;
    errorbar(x_values, pred(:, i), err(:, i), 'r');
    title(titles(i), 'fontsize', 20);
end

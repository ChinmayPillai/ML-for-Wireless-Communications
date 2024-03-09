%% Cleaning Up
% Close all previously opened figueres and clear workspace
close all;
clear;


%% Part A - Calculating 't'
% Initialise coefficients, N and xMax
w0 = 1;
w1 = -2;
w2 = 0.5;
N = 200;
xMax= 5;

% Generate random noise column vector from standard normal distribution
n = randn(N, 1);

% Generate x from uniform distributrion from -xMax to +xMax
x = 2*xMax*(rand(N,1) - 0.5);
% Sort x
x = sort(x);

% Find t vector
t = w0 + w1*x + w2*(x.^2) + n;


%% Plotting 't'
figure(1);
% Plot generated data
plot(x,t,'k.','markersize', 8);
xlim([-xMax xMax])
xlabel('$x$','interpreter','latex','fontsize',15);
ylabel('$t$','interpreter','latex','fontsize',15);
hold on


%% Part B - Fitting Least Square Models
% Least Square solution is 
% w = (X'X)^-1 * X' * t (X' is X transpose)

% Generate feature matrix X from x
% [1 x x^2]
X = [];

% Loop through k
for k = 0:2
    % Append x^k column to X
    X = [X x.^k];

    if k == 1
        % For linear fit we want X = [1 x]
        w_linear = inv(X'*X) * X' * t;
    end
end

% For quadratic fit we want X = [1 x x^2]
w_quad = inv(X'*X) * X' * t;


%% Plotting least square model
% Generate feature matrix for points to predict
x_values = [-1*xMax:0.01:xMax]';
features = [];

for k = 0:2
    features = [features x_values.^k];
end

% Plot linear least square fit at (x_values, w_linear*features)
% Here take only first 2 columns of features - [1 x]
plot(x_values, features(:,1:2)*w_linear,'g','linewidth',2);

% Plot quadratic least square fit at (x_values, w_quad*features)
% Here take all 3 columns of features - [1 x x^2]
plot(x_values, features*w_quad,'r','linewidth',2);
legend('Data','Linear','Quadratic')
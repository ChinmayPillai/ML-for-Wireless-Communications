%% Cleaning Up
% Close all previously opened figueres and clear workspace
close all;
clear;


%% Part A - Calculating 't'
% Initialise coefficients, N and order
w0 = -3;
w1 = 2;
N = 6;
order = 5;

% Generate random noise column vector from standard normal distribution
variance = 3;
n = sqrt(variance) * randn(N, 1);

% Generate x uniformly spread between 0 and 1
x = [0:1/(N-1):1]';

% Find t vector
t = w0 + w1*x + n;


%% Plotting 't'
figure(1);
% Plot generated data
plot(x, t, 'b.', 'markersize', 25);
xlabel('$x$','interpreter','latex','fontsize',15);
ylabel('$t$','interpreter','latex','fontsize',15);
legend('Location', 'Northwest');
legend('Data');
hold on


%% Part B - Fitting & Plotting 5th order Regularised Least Square Models

% Generate feature matrix X from x
% [1 x x^2 x^3 x^4 x^5]
X = [];

% Loop through k
for k = 0:order
    % Append x^k column to X
    X = [X x.^k];
end

lambda_arr = [0, 1e-6, 0.01, 0.1];
% Generate feature matrix for points to predict
x_values = [0:0.01:1]';
features = [];

for k = 0:order
    features = [features x_values.^k];
end

% Regularised Least Square solution is 
% w = (X'X + N*lamda*I)^-1 * X' * t 
% where (X' is X transpose) & (I is identity matrix)

% For all lamdas
for i = 1:length(lambda_arr)
    lambda = lambda_arr(i);
    
    % Compute regularised least square weight
    w = inv(X'*X + N*lambda*eye(order+1)) * X' * t;

    % Plot the reg. least sq. wt. for a particular lambda
    figure(i+1);
    plot(x,t,'b.','markersize',20);
    hold on;
    plot(x_values, features*w,'r','linewidth',2)
    xlim([-0.1 1.1])
    xlabel('$x$','interpreter','latex','fontsize',15);
    ylabel('$t$','interpreter','latex','fontsize',15);
    ti = sprintf('$\\lambda = %g$',lambda);
    title(ti,'interpreter','latex','fontsize',20)
    legend('Location', 'Northwest');
    legend('Data', 'Model');
end

%% Part C - Verify behaviour

% We can observe from the plots that for 
% lambda = 0 - Model is exactly over-fitting the 6 data points
% lambda = 1e-6 - Model fits the general shape of a 5th order polynomial
%                 but with less variation, complexity and hence is further 
%                 the data points
% lambda = 0.01 - Model is simpler, with lower variations and resembles a 
%                 2nd/3rd order polynomial more than a 5th order one. 
%                 The model is even less complex and hence also has higher 
%                 error for the points. It doesn't over-fit the data
% lambda = 0.1 - Model is much more simple and had extremely low variation 
%                in slope. It also doesn't over-fit the data
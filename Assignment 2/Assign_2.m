%% Sparse Bayesian Learning for Regression
% EE798L: Machine Learning for Wireless Communications
% Assignment 2

clear; clc;
close all;
warning('off');


%% Parameters to Generate Synthetic Data
N = 20;                                     % Number of data points
M = 40;                                     % Number of basis functions
D0 = 7;                                     % Number of non-zero weights

% Generate design matrix Phi
Phi = randn(N, M);

% Generate the sparse weight vector
w = zeros(M, 1);                            % Initialise w to 0
nonzero_indices = randperm(M, D0);          % D0 random indices in [1,M]
% Assign random value to selected indices from standard Gaussian dist.
w(nonzero_indices) = randn(D0, 1);           

% Generate noise variances
noise_variances = [-20, -15, -10, -5, 0];   % in dB
lin_var = 10.^(noise_variances./10);        % Convert dB to linear scale

% Initialise nmse to 0
nmse_values = zeros(1, length(noise_variances));

%% Run algorithm several times and obtain average NMSE
       
% Set iteration conditions
iterations = 2000;                              % No. of iterations
convergence_threshold = 5e-3;                   % Convergenge threshhold
% Max iterations with no convergence allowed
max_conv_iterations = 100;      


% Run algorithm 'iterations' number of times
for iteration = 1:iterations
    
    % Run algorithm for each noise_variance value
    for i = 1:length(noise_variances)
       %% Generate the observations
       epsilon = sqrt(lin_var(i)) * randn(N, 1);    % Noise vector
       t = Phi * w + epsilon; % Observations
       
       %% Iterative Alteranate Optimization Algorithm


       % Initialize the parameters
       alpha = 1e-6 * ones(M, 1);               % Initial hyperparameters
       beta = 1e-6;                             % Initial noise precision

       % alpha_old = -100;
       % beta_old = -100;
       mu_old = ones(M,1);

       
       for iter = 1:max_conv_iterations
           % Update the posterior weight distribution
           A = diag(alpha);
           Sigma = inv(beta * (Phi' * Phi) + A);
           mu = Sigma * Phi' * (beta * t);
           
           % Update the hyperparameters
           gamma = 1 - alpha .* diag(Sigma);
           alpha = gamma ./ (mu.^2);
           beta = (N - sum(gamma)) / norm(t - Phi * mu)^2;
           
           % Check for convergence
           
           % alpha_change = sum(abs(alpha_old - alpha)) / sum(alpha);
           % beta_change = abs(beta_old - beta) / beta;
           
           rel_mu_change = abs(norm(mu-mu_old)/norm(mu_old));
           if (rel_mu_change <= convergence_threshold)
               %disp("Convergence Achieved")
               break;
           end

           % alpha_old = alpha;
           % beta_old = beta;
           mu_old = mu;
       end
       
       % Obtain the maximum a posteriori (MAP) estimate of the weight vector
       w_mp = mu;
       
       % Calculate and aggregate the NMSE for each noise variance
       nmse_values(i) = nmse_values(i) + norm(w_mp - w)^2 / norm(w)^2;
    end
    fprintf('Progress: %d%%\n', round(iteration / iterations * 100));
end

% Get average NMSE
nmse_values = nmse_values./iterations;
% Convert NMSE to dB for comparision with noise variance
nmse_db = 10 * log10(nmse_values);


%% Plot NMSE vs Noise Variance

% Plot the NMSE versus noise variance
figure;
plot(noise_variances, nmse_values, '-o');
grid on;
xlabel('Noise Variance (dB)');
ylabel('Normalized Mean Squared Error (NMSE)');
title('NMSE vs. Noise Variance');

% Plot the NMSE (dB) versus noise variance
figure;
plot(noise_variances, nmse_db, '-o');
grid on;
xlabel('Noise Variance (dB)');
ylabel('Normalized Mean Squared Error (NMSE)(dB)');
title('NMSE (dB) vs. Noise Variance');
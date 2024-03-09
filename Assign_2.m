%% Sparse Bayesian Learning for Regression
% EE798L: Machine Learning for Wireless Communications
% Assignment 2

% Generate Synthetic Data
clear; clc;

N = 20; % Number of data points
M = 40; % Number of basis functions
D0 = 7; % Number of non-zero weights

% Generate design matrix Phi
Phi = randn(N, M);

% Generate the sparse weight vector
w = zeros(M, 1);
nonzero_indices = randperm(M, D0);
w(nonzero_indices) = randn(D0, 1);

% Generate noise variances
noise_variances = [-20, -15, -10, -5, 0]; % in dB

% Generate the observations
nmse_values = zeros(1, length(noise_variances));
for i = 1:length(noise_variances)
   sigma2 = 10^(noise_variances(i)/10); % Convert dB to linear scale
   epsilon = sqrt(sigma2) * randn(N, 1); % Noise vector
   t = Phi * w + epsilon; % Observations

   % Step 4: Apply SBL for regression
   
   % Initialize the parameters
   alpha = 1e-6 * ones(M, 1); % Initial hyperparameters
   beta = 1e-6; % Initial noise precision
   
   % Iterative optimization
   convergence_threshold = 1e-6;
   max_iterations = 1000;
   alpha_old = -100;
   beta_old = -100;
   for iter = 1:max_iterations
       % Update the posterior weight distribution
       A = diag(alpha);
       Sigma = inv(beta * (Phi' * Phi) + A);
       mu = Sigma * Phi' * (beta * t);
       
       % Update the hyperparameters
       gamma = 1 - alpha .* diag(Sigma);
       alpha = gamma ./ (mu.^2);
       beta = (N - sum(gamma)) / norm(t - Phi * mu)^2;
       
       % Check for convergence
       alpha_change = sum(abs(alpha_old - alpha)) / sum(alpha);
       beta_change = abs(beta_old - beta) / beta;
       if (abs(alpha_change) < convergence_threshold) && (abs(beta_change) < convergence_threshold)
           disp("Convergence Achieved")
           break;
       end
       alpha_old = alpha;
       beta_old = beta;

   end
   
   % Obtain the maximum a posteriori (MAP) estimate of the weight vector
   w_mp = mu;
   
   % Step 5: Calculate and plot the NMSE
   nmse_values(i) = norm(w_mp - w)^2 / norm(w)^2;
end

% Plot the NMSE versus noise variance
figure;
plot(noise_variances, nmse_values, '-o');
xlabel('Noise Variance (dB)');
ylabel('Normalized Mean Squared Error (NMSE)');
title('NMSE vs. Noise Variance');
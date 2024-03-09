%% Cleaning Up
% Close all previously opened figueres and clear workspace
close all;
clear;


%% Hyper Parameters and normalizing constants

mean = [1;2];
% Covariance matrices for a and b part
cov1 = [1 0; 0 1];
cov2 = [1 0.8;0.8 1];

% Normalizing Constants for a and b parts
const1 = 1/(2*pi*sqrt(det(cov1)));
const2 = 1/(2*pi*sqrt(det(cov2)));

% Combine the covariance matrices into a single 3D matrix
cov(:,:,1) = cov1;
cov(:,:,2) = cov2;
% Combine the normalizing constants into a single vector
const = [const1 const2];


%% Computing and Plotting
% Create Mesh Grid for plot
[X,Y] = meshgrid(-5+mean(1):0.1:5+mean(1),-5+mean(2):0.1:5+mean(2));

% Find x and y differences of each point from mean
diff = [X(:)-mean(1) Y(:)-mean(2)];

% For part a and part b
for i = 1:2
    % Calculate pdf
    pdfv = const(i)*exp(-0.5*diag(diff*inv(cov(:,:,i))*diff'));
    pdfv = reshape(pdfv,size(X));
    
    % Plot Contour
    figure(2*i - 1);
    hold off;
    contour(X,Y,pdfv);
    % Plot Surface
    figure(2*i);
    hold off;
    surf(X,Y,pdfv);
end
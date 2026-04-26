% =========================================================================
% Improved GDHLDA for Linear Dimensionality Reduction on the Stiefel Manifold
% =========================================================================
%
% This script implements an improved version of GDHLDA
% (gradient descent-based multiclass harmonic linear discriminant analysis)
% for supervised dimensionality reduction. This method computes an
% orthonormal basis W on the Stiefel manifold by minimizing a harmonic
% trace-ratio objective that combines a harmonic within-class scatter
% matrix with pairwise between-class scatter matrices.
%
% The implementation includes the main algorithmic improvements introduced
% in the improved GDHLDA framework:
%   - random initialization
%   - Riemannian gradient normalization
%   - adaptive step size with random relaxation
%   - orthogonality-based conditional retraction
%   - QR/SVD retraction options
%
% Input:
%   A CSV file containing feature vectors and class labels
%
% Output:
%   W      - projection matrix with orthonormal columns
%   Jvals  - objective values across iterations
%
% This code was written to reproduce the improved GDHLDA procedure
% described in the following work:
%
%   Oh, M., Oh, C., and Tabassum, E.
%   "Covalent: Interpretable and Discriminative Collective Variables
%   Reveal Ligand-Dependent Switching in Human Cellular Retinol-Binding
%   Protein 2"
%   Journal of Chemical Theory and Computation, 2025
%   DOI: 10.1021/acs.jctc.5c01402
% =========================================================================
clc; clear; close;
format long

%% User Setting
r = 2;              % Number of eigenvectors
eta = 1e-3;         % Initial step size
q = inf;            % Upper bound on step size growth
a = -0.9;           % Lower bound of relaxation parameter (a > -1)
b = 1.5;            % Upper bound of relaxation parameter (b > a)
stol = eps();       % Division tolerance
otol = eps();       % Orthogonality criterion
ftol = eps();       % Convergence criterion (function value)
gtol = eps();       % Convergence criterion (gradient)
etamin = eps();     % Convergence criterion (step size)
miniter = 100;      % Minimum convergence criterion
maxiter = 1000;     % Maximum convergence criterion

%% Helper Functions
symm = @(W) 0.5*(W + W');           % symmetric matrix
proj = @(W, Z) Z - W*symm(W'*Z);    % projection of Z to tangent space at W

%% Dataset Preprocessing
% Load dataset from uploaded file
T = readtable('example.csv');
X = T{:, 1:3};  % features
Y = T{:, 4};    % labels

% Find size of dataset (number of samples and number of features)
[n, p] = size(X);

% Find unique classes and number of classes
c = unique(Y);
nc = length(c);

% Set empty arrays for class sizes and class mean vectors
ncs = zeros(nc, 1);
mu = zeros(nc, p);

% Compute Sw with size of class dataset and mean vectors
Sw = zeros(p);
for k = 1:nc
    ind = find(Y == c(k));
    ncs(k) = length(ind);           % class sizes
    mu(k, :) = mean(X(ind, :), 1);  % class mean vectors

    Sk = zeros(p);
    for i = 1:ncs(k)
        Xc = X(ind(i), :) - mu(k, :);
        Sk = Sk + Xc'*Xc;
    end
    Sw = Sw + pinv(Sk);
end
Sw = pinv(Sw);

%% Riemmanian Gradient Descent
% Initialize W on Stiefel
W = retr(randn(p, r), zeros(p, r), "QR");

% Preallocate Jvals to record function values
Jvals = nan(maxiter+1, 1);

% Store initial function value
Jvals(1) = J(W, nc, ncs, mu, Sw, stol);

% Compute initial Euclidean gradient
grad = zeros(p, r);
trSw = trace(W'*Sw*W);
for k = 1:nc-1
    for l = k+1:nc
        d = mu(k, :) - mu(l, :);
        Bkl = d'*d;
        trBkl = trace(W'*Bkl*W);
        grad = grad + 2*ncs(k)*ncs(l)*(trBkl*Sw*W - trSw*Bkl*W) / max(trBkl^2, stol);
    end
end

% Compute and normalize initial Riemannian gradient
xi = proj(W, grad);
xih = xi/max(norm(xi, 'fro'), stol);

% Main loop
for counter = 1:maxiter
    % Store old state
    xio = xi;               % old Riemannian gradient
    xiho = xih;             % old normalized Riemannian gradient
    Jo = Jvals(counter);    % old function value

    % Update W on tangent space using gradient descent method
    Wtrial = W - eta*xih;

    % Check orthogonality for retraction
    org = norm(Wtrial'*Wtrial - eye(r), 'fro') / norm(eye(r), 'fro');
    if org > otol
        W = retr(W, -eta*xih, "QR");
    else
        W = Wtrial;
    end

    % Store current objective function value
    Jw = J(W, nc, ncs, mu, Sw, stol);
    Jvals(counter+1) = Jw;
    
    % Display progress
    if mod(counter, 50) == 0
        fprintf('Iteration: %d, J: %.15e, Stepsize: %.5e\n', counter, Jw, eta);
    end

    % Compute new Euclidean gradient
    grad = zeros(p, r);
    trSw = trace(W'*Sw*W);
    for k = 1:nc-1
        for l = k+1:nc
            d = mu(k, :) - mu(l, :);
            Bkl = d'*d;
            trBkl = trace(W'*Bkl*W);
            grad = grad + 2*ncs(k)*ncs(l)*(trBkl*Sw*W - trSw*Bkl*W) / max(trBkl^2, stol);
        end
    end
    
    % Normalize new Riemannian gradient
    xi = proj(W, grad);
    xih = xi/max(norm(xi, 'fro'), stol);

    % Perform (extrinsic) vector transport via projection transport for step adaptation
    xioT = proj(W, xio);    % transport old Riemannian gradient to T_W
    xihoT = proj(W, xiho);  % transport old normalized Riemannian gradient to T_W
    
    % Perform random step adaptation with relaxation
    alpha = a+(b-a)*rand;
    numerator = (1+alpha)*abs(trace(xioT'*xihoT));
    denominator = abs(trace(xioT'*xihoT) - trace(xi'*xihoT));
    denominator = max(denominator, stol);

    if  numerator > q*denominator
        eta = sqrt(q)*eta;
    else
        eta = sqrt(numerator / denominator)*eta;
    end
    
    % Check convergence
    if counter >= miniter
        if abs(Jw - Jo)/max(1, abs(Jo)) < ftol
            break
        elseif norm(xi, 'fro') < gtol
            break
        elseif eta < etamin
            break
        end
    end
end

% Display fianl results
fprintf('[Iteration] %d, [J] %.15e, [Stepsize] %.5e\n', counter, Jw, eta);

% Trim Jvals to actual length
Jvals = Jvals(1:counter+1);

%% Results
% Export W
% save('W', 'W')

% Draw convergence curve
figure(1)
plot(Jvals, 'LineWidth', 2.5)
xlabel("Iteration")
ylabel("Function Value")
title("Convergence Curve")
grid on

%% Important Functions
% Define objective function
function f = J(W, nc, ncs, mu, Sw, stol)
f = 0;
trSw = trace(W'*Sw*W);
for k = 1:nc-1
    for l = k+1:nc
        Bkl = (mu(k, :) - mu(l, :))'*(mu(k, :) - mu(l, :));
        trBkl = trace(W'*Bkl*W);
        f = f + ncs(k)*ncs(l)*trSw / max(trBkl, stol);
    end
end
end

function X = retr(W, xi, flag)
% QR is usually faster than SVD when r << p
    if nargin < 3 || isempty(flag)
        flag = "SVD";   % default choice
    end
    flag = upper(string(flag));

    if flag == "QR"
        % QR retraction (fast)
        [Q, R] = qr(W + xi, 'econ');
        S = sign(diag(R));
        S(S == 0) = 1;
        X = Q * diag(S);
    elseif flag == "SVD"
        % SVD retraction (robust)
        [U, ~, V] = svd(W + xi, 'econ');
        X = U*V';
    else
        error('retr:Unknown', 'flag must be "SVD" or "QR". Got "%s".', flag);
    end
end
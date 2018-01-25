function [theta, accuracy] = logisticRegression(lambda)

data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

X = mapFeature(X(:,1), X(:,2));

theta = zeros(size(X, 2), 1);

options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), theta, options);

plotDecisionBoundary(theta, X, y);

p = predict(theta, X);

accuracy = mean(double(p == y)) * 100

function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 

figure; hold on;

positiveCases = find(y == 1);
negativeCases = find(y == 0);

plot(X(positiveCases, 1), X(positiveCases, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(negativeCases, 1), X(negativeCases, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

hold off;

end

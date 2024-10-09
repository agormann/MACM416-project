function tiledPlot = PlotPDE(u, x, t, N)
%PLOTPDE Summary of this function goes here
%   Detailed explanation goes here

minT = min(t); maxT = max(t);
[X, T] = meshgrid(x,t);
U = u(X,T);

tiledPlot = tiledlayout(1, 2);

nexttile;
view(3);
xlabel('x'); ylabel('t'); zlabel('u(x,t)');
grid on;

hold on;
surf(X, T, U, 'EdgeColor', 'none');
hold off;

nexttile;
view(2);
legend('Location', 'best');
xlabel('x'); ylabel('u(x,t)');
grid on;
C = hsv(N);

hold on;
formatSpec = "u(x, %d)";
n = (maxT-minT)/(N-1);
for i = 0:N-1
    name = sprintf(formatSpec, n);
    plot(x, u(x, n*ones(1, length(t))), 'color', C(i+1, :),...
        'DisplayName', name);
    n = n+1;
end
hold off;

end


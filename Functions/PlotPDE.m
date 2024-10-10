function tiledPlot = PlotPDE(u, x, t, N)
%PLOTPDE Summary of this function goes here
%   Detailed explanation goes here

xMin = min(x); xMax = max(x);
tMin = min(t); tMax = max(t);
[X, T] = meshgrid(x,t);
U = double(u(X,T));

tiledPlot = tiledlayout(1, 3);

% x vs. t vs. u(x,t)
nexttile;
view(3);
xlabel('x'); ylabel('t'); zlabel('u(x,t)');
grid on;

hold on;
surf(X, T, U, 'EdgeColor', 'none');
hold off;

% x vs. u(x,t)
nexttile;
view(2);
legend('Location', 'best');
xlabel('x'); ylabel('u(x,t)');
grid on;
C = hsv(N);

hold on;
formatSpec = "u(x, %.2f)";
inc = (tMax-tMin)/(N-1);
n = 0;
for i = 0:N-1
    name = sprintf(formatSpec, n);
    plot(x, u(x, n*ones(1, length(t))), 'color', C(i+1, :),...
        'DisplayName', name);
    n = n+inc;
end
hold off;

% x vs. t
x0 = linspace(xMin, xMax, N);

nexttile;
view(2);
xlabel('x'); ylabel('t');
grid on;
C = hsv(N);
axis([xMin xMax tMin tMax]);

hold on;
for i = 1:N
    m = double(u(x0(i),0));
    plot(x, (x-x0(i))/(m+eps), 'color', C(i, :));
end
hold off;

end


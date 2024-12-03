function U = riemann(uL,uR,X,T)
%RIEMANN Solves a riemann problem exactly.
% Input
%   uL := value of profile to the left of 0
%   uR := value of profile to the right of 0
%   X := grid of x values
%   T := specific times to solve the problem at
% Output
%   U := matrix whose rows are the solution to the riemann problem at times specified in T.

nx = length(X);
u0 = uL*(X<0) + uR*(X>=0);

U = [u0; zeros(length(T)-1,nx)];

if uL > uR
    % shockwave
    s = (uL+uR)/2;
    for i = 2:length(T)
        t = T(i);
        U(i,:) = uL*(X<s*t) + uR*(X>=s*t);
    end
elseif uL < uR
    % rarefaction wave
    for i = 2:length(T)
        t = T(i);
        U(i,:) = uL*(X<uL*t) + (X/t).*((uL*t<=X)&(X<=uR*t)) + uR*(X>uR*t);
    end
end

end


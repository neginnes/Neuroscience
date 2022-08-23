function Y = simple_model2(B,sigma,dt,timeInterval,X)
    mu = B*timeInterval;
    sigma = sqrt(length(0:dt:timeInterval) * sigma^2 * dt*2);
    y = cdf('Normal',X,mu,sigma);
    p = rand(1);
    if p < y 
        Y = 0;
    else
        Y = 1;
    end
end

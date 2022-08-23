function [Y,Xvalues] = simple_model(B,sigma,dt,timeInterval,X)
    Xvalues = X;  
    for t = 0:dt:timeInterval-dt
        dW = normrnd(0,dt);
        dX = B*dt + sigma*dW;
        X = X + dX;
        Xvalues = [Xvalues;X];
    end

    if  Xvalues(end) >= 0
        Y = 1;
    else
        Y = 0;
    end  
end

function [Y,Xvalues,Xvalues1,Xvalues2,T] = race_trial(B1,B2,sigma1,sigma2,dt,th1n,th2p,X1,X2)
    Xvalues1 = X1;
    Xvalues2 = X2;
    T = 0;
    while (1)
        dW1 = normrnd(0,dt);
        dW2 = normrnd(0,dt);
        dX1 = B1*dt + sigma1*dW1;
        dX2 = B2*dt + sigma2*dW2;
        X1 = X1 + dX1;
        X2 = X2 + dX2;
        T = T + dt;
        Xvalues1 = [Xvalues1;X1];
        Xvalues2 = [Xvalues2;X2];
        if X2 >= th2p
            Y = 1;
            Xvalues = Xvalues2;
            break;
        elseif X1 <= th1n
            Y = 0;
            Xvalues = Xvalues1;
            break
        end
    end
end
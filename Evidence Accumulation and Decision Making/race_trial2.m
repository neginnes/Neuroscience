function [Y,Xvalues,Xvalues1,Xvalues2,T] = race_trial2(B1,B2,sigma1,sigma2,dt,th1n,th2p,X1,X2,timeInterval)
    Xvalues1 = X1;
    Xvalues2 = X2;
    T = 0;
    flag = 0;
    while (T<timeInterval)
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
            flag = 0;
            break;
        elseif X1 <= th1n
            Y = 0;
            Xvalues = Xvalues1;
            flag = 0;
            break
        end
    end
    if (flag == 0)
        if (abs(th1n-X1)< abs(th2p-X2))
            Y = 0;
            Xvalues = Xvalues1;
        else 
            Y = 1;
            Xvalues = Xvalues2;
        end
    end
end
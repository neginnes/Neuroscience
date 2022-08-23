function [Y, Xvalues, T] = two_choice_trial(B,sigma,dt,thn,thp,X)
    Xvalues = X;
    T = 0;
    while (1)
        dW = normrnd(0,dt);
        dX = B*dt + sigma*dW;
        X = X + dX;
        T = T + dt;
        Xvalues = [Xvalues;X];
        if X >= thp
            Y = 1;
            break;
        elseif X <= thn
            Y = 0;
            break
        end
    end
end

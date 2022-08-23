function [act,acti,actj] = takeAction(probs,acts)
    nActs = length(probs);
    p = rand(1);
    for i = 1 : nActs
        if(p >= sum(probs(1:i-1)) && p < sum(probs(1:i)))
            act = acts(i);
        end
    end
    if (act == 1)
        acti = 1;
        actj = 0;
    elseif (act == 2)
        acti = 0;
        actj = -1;
    elseif (act == 3)
        acti = -1;
        actj = 0;
    elseif (act == 4)
        acti = 0;
        actj = 1;
    end
end
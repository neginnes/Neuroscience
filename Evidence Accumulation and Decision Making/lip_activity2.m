function [MT,LIP,LIP_event_times1,LIP_event_times2,rates1,rates2,tVec] = lip_activity2(MT_p_values,LIP_weights,LIP_threshold,M)
    t = 0;
    dt = 0.001;
    MT = [0 0];
    LIP = [];
    N = MT; 
    LIP_event_times1 = [];
    LIP_event_times2 = [];
    N_LIP1 = 0;
    N_LIP2 = 0;
    rate1 = 0;
    rate2 = 0;
    rates1 = rate1;
    rates2 = rate2;
    tVec = t;
    cnt = 1;
    while (round(rate1 - LIP_threshold(1))<0 & round(rate1 - LIP_threshold(2))<0 & t<0.5)
        if(t>0.5)
            break
        end
        t = t + dt;
        dN = rand(1, 2) < MT_p_values(:,cnt)';   
        MT = [MT;dN];
        N = N + dN;
        p_LIP1 = sum(N.*LIP_weights);
        p_LIP2 = sum(N.*[LIP_weights(2),LIP_weights(1)]);
        LIP_event = [rand(1) < p_LIP1, rand(1) < p_LIP2];
        LIP = [LIP;LIP_event];
        if (LIP_event(1) == 1)
            LIP_event_times1 = [LIP_event_times1,t];
            N_LIP1 = N_LIP1 + 1;
        end
        if (LIP_event(2) == 1)
            LIP_event_times2 = [LIP_event_times2,t];
            N_LIP2 = N_LIP2 + 1;
        end
        if (N_LIP1 > M)
            rate1 = M/(t-LIP_event_times1(N_LIP1-M));
        end
        if (N_LIP2 > M)
            rate2 = M/(t-LIP_event_times2(N_LIP2-M));
        end
        rates1 = [rates1,rate1];
        rates2 = [rates2,rate2];
        tVec = [tVec,t];
        cnt = cnt + 1;
    end
end
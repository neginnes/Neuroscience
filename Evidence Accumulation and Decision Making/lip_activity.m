function [MT,LIP,LIP_event_times,rates,tVec] = lip_activity(MT_p_values,LIP_weights,LIP_threshold,M)
    t = 0;
    dt = 0.001;
    MT = [0 0];
    LIP = 0;
    N = MT; 
    LIP_event_times = [];
    N_LIP = 0;
    rate = 0;
    rates = rate;
    tVec = t;
    while (round(rate-LIP_threshold)<0)
        if(t>0.5)
            break
        end
        t = t + dt;
        dN = rand(1, 2) < MT_p_values;     
        MT = [MT;dN];
        N = N + dN;
        p_LIP = sum(N.*LIP_weights);
        LIP_event = rand(1) < p_LIP;
        LIP = [LIP,LIP_event];
        if (LIP_event == 1)
            LIP_event_times = [LIP_event_times,t];
            N_LIP = N_LIP + 1;
        end
        if (N_LIP > M)
            rate = M/(t-LIP_event_times(N_LIP-M));
        end
        rates = [rates,rate];
        tVec = [tVec,t];
    end
end
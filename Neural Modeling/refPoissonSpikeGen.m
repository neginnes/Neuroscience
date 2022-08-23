function SpikeTimes = refPoissonSpikeGen(lambda, tSim, refPeriod)
    SpikeTimes = [];  
    t = 0;
    while (t < tSim)
        t = t + exprnd(1/lambda) + refPeriod;
        SpikeTimes = [SpikeTimes,t];
    end
end
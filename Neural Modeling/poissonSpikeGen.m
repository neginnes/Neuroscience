function [spikeMat, tVec] = poissonSpikeGen(lambda, tSim, nTrials)
    dt = 1/1000;
    nBins = floor(tSim/dt);
    spikeMat = rand(nTrials, nBins) < lambda*dt;
    tVec = 0:dt:tSim-dt;
end
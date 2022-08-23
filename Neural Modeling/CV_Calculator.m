function CV = CV_Calculator(spikeMat,tVec)
    CV = [];
    for trialCount = 1:size(spikeMat,1)
        spike_intervals = [];
        spikePos = tVec(spikeMat(trialCount, :));
        spike_intervals = [spike_intervals, diff(spikePos)];
        if (length(spike_intervals) ~= 0)
            CV = [CV,std(spike_intervals)/(0.0000000000000001 + mean(spike_intervals))];
        else
            CV = [CV,0];
        end
        
    end
end
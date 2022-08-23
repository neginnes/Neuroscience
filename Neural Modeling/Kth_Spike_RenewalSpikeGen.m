function spikeMat = Kth_Spike_RenewalSpikeGen(input_spikeMat, k)
    spikeMat = input_spikeMat;
    for trialCount = 1:size(input_spikeMat,1)
        c = 0;
        for spikeCount = 1:size(input_spikeMat,2)
            if (input_spikeMat(trialCount,spikeCount) > 0)
                c = c + 1;
            end
            if (c < k)
                spikeMat(trialCount,spikeCount) = 0;
            end
            if (c == k)
                c = 0;
            end
        end
    end
end
function output_spikeTrain = EI_CoincidenceDitector(spikeTrain,Nnet,D)
    dt = 1/1000;
    W = round(D/dt);
    i = 1;
    output_spikeTrain = zeros(size(spikeTrain));
    while i <= length(spikeTrain)-W    
        spikeWindow = spikeTrain(i : i+W);
        Ne = length(find(spikeWindow==1));
        Ni = length(find(spikeWindow==-1));
        if (Ne-Ni >= Nnet)
            output_spikeTrain(i+W) = 1;
        end       
        i = i + 1;    
    end
end
function output_spikeTrain = CoincidenceDitector(spikeTrain,Ratio,D)
    dt = 1/1000;
    W = round(D/dt);
    M = length(find(spikeTrain==1));
    i = 1;
    output_spikeTrain = zeros(size(spikeTrain));
    while i <= length(spikeTrain)-W    
        spikeWindow = spikeTrain(i : i+W);
        N = length(find(spikeWindow==1));
        if (N/M >= Ratio)
            output_spikeTrain(i+W) = 1;
        end       
        i = i + 1;    
    end
end
    
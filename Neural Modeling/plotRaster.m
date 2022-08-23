function plotRaster(spikeMat, tVec, Title)
    hold all;
    for trialCount = 1:size(spikeMat,1)
        spikePos = tVec(spikeMat(trialCount, :)==1);
        for spikeCount = 1:length(spikePos)
            plot([spikePos(spikeCount), spikePos(spikeCount)], [trialCount-0.4,trialCount+0.4], 'k');
        end
    end
    ylim([0, size(spikeMat, 1)+1]);
    title(Title,'interpreter','latex')
    xlabel('Time(s)','interpreter','latex')
    ylabel("Trial number",'interpreter','latex')
end
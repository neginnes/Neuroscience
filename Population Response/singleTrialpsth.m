function MAT = singleTrialpsth(data,window_width,unit_number,start_time,stop_time,figure_flag)
    nBins = (stop_time-start_time)/window_width;
    MAT = zeros(length(data),nBins);
    spikes = data;
    for j = 1:nBins
        afterSpikes = spikes(find(spikes > (j-1)*window_width + start_time));
        MAT(j) = length(find(afterSpikes <= j*window_width + start_time));
    end
    MAT = mean(MAT,1)/window_width;
    y = smooth(MAT);
    if (figure_flag == 1)
        plot(start_time + window_width/2 :window_width:stop_time - window_width/2 ,y);
        Y_MAX = 1.1*max(y);
        ylim([0,Y_MAX]);
        xlim([start_time,stop_time]);
        hold on
        plot([0,0],[0,Y_MAX]);
        legend('PSTH','Cue Onset','interpreter','latex')
        xlabel('t(s)','interpreter','latex');
        ylabel('PSTH','interpreter','latex');
        title("unit number  = " + num2str(unit_number),'interpreter','latex');
    end
end
 
 

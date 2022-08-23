function [weights, sigmas,sigmaMat] = KalmanFilter(tauw, tauv, paradigm, preTrainingTrialNumber, trainingTrialNumber, sigma, plottingFlag)
    weights = [];
    sigmas = [];
    sigmaMat = [];
    w = [0, 0];
    A = eye(2);
    for i = 1 : preTrainingTrialNumber
        weights = [weights; w];
        sigmas = [sigmas; [sigma(1, 1), sigma(2, 2)]];
        sigmaMat = [sigmaMat, [sigma(1,1);sigma(2,2);sigma(1,2);sigma(2,1)]];
        if isequal(paradigm, 'backward blocking')
            C = [1;1];
            r = 1;
            Wnoise = tauw^2;
            Vnoise = tauv^2;
        else
            C = [1;0];
            r = 1;
            Wnoise = [tauw^2,0];
            Vnoise = tauv^2;
        end
        y = r;
        w = w*A;
        sigma = A*sigma*A' + Wnoise;
        G = sigma*C*((C'*sigma*C + Vnoise)^-1);
        sigma = sigma - G*C'*sigma;
        w = w + G'*(y - w*C);
    end
    for i = 1 : trainingTrialNumber 
        weights = [weights; w];
        sigmas = [sigmas; [sigma(1, 1), sigma(2, 2)]];
        sigmaMat = [sigmaMat, [sigma(1,1);sigma(2,2);sigma(1,2);sigma(2,1)]];
        Wnoise = tauw^2;
        Vnoise = tauv^2;
        if isequal(paradigm, 'blocking')
            C = [1;1];
            r = 1;
        elseif isequal(paradigm, 'unblocking')
            C = [1;1];
            r = 2;
        elseif isequal(paradigm, 'backward blocking')
            C = [1;0];
            r = 1;
        end
        y = r;
        w = w*A;
        sigma = A*sigma*A' + Wnoise;
        G = sigma*C*((C'*sigma*C + Vnoise)^-1);

        sigma = sigma - G*C'*sigma;
        w = w + G'*(y - w*C); 

    end
    

    trials = 0 : trainingTrialNumber + preTrainingTrialNumber - 1;
    trainingTrials = preTrainingTrialNumber : trainingTrialNumber + preTrainingTrialNumber-1;
    if isequal(plottingFlag, 'True')
        set(gcf,'color','w');
        subplot(2, 1, 1);
        plot(trials,weights(:, 1));
        hold on
        plot(trials,weights(:, 2));
        hold on
        Min = min(weights,[],'all');
        Max = 1.1*max(weights,[],'all');
        ylim([Min,Max]);
        plot(ones(1,size(weights,1))*10, linspace(Min,Max,size(weights,1)), 'x','Color','black');
        xlabel('trial number','interpreter', 'latex');
        ylabel('w(t)','interpreter', 'latex');
        legend('$$w_1$$', '$$w_2$$','interpreter', 'latex');
        title(num2str(paradigm) + " weights",'interpreter', 'latex');

        subplot(2, 1, 2);
        plot(trials,sigmas(:, 1));
        hold on
        if isequal(paradigm, 'backward blocking')
            plot(trials,sigmas(:, 2));            
        else
            plot(trainingTrials,sigmas(trainingTrials+1, 2));
        end
        hold on
        Min = min(sigmas,[],'all');
        Max = max(sigmas,[],'all');
        ylim([Min,Max]);
        plot(ones(1,size(sigmas,1))*10, linspace(Min,Max,size(sigmas,1)), 'x','Color','black');
        xlabel('trial number','interpreter', 'latex');
        ylabel('$$\sigma^2$$(t)','interpreter', 'latex');
        legend('$$w_1$$', '$$w_2$$','interpreter', 'latex');
        title(num2str(paradigm) + " sigmas",'interpreter', 'latex');
    end
end
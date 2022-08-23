%% Part1.1
clear all
close all
clc

addpath(genpath('E:\neuro\Eye tracking database'))
datafolder = 'Eye tracking database\Eye tracking database\Eye tracking database\DATA\hp';
stimfolder = 'Eye tracking database\Eye tracking database\Eye tracking database\ALLSTIMULI';

files=dir(fullfile(datafolder,'*.mat'));
[filenames{1:size(files,1)}] = deal(files.name);
Nstimuli = randperm(size(filenames,2),4);

showEyeData(datafolder,stimfolder,Nstimuli);


%% Part1.2
close all
clc
addpath(genpath('E:\neuro\Eye tracking database'))
set(gcf,'Color','w')
showEyeDataAcrossUsers('Eye tracking database\Eye tracking database\Eye tracking database\ALLSTIMULI',7,Nstimuli);
set(gcf,'Color','w')

%% Part 2.1
clear all
close all
clc
addpath(genpath('E:\neuro\Eye tracking database'))
image = 'E:\neuro\Eye tracking database\Eye tracking database\Eye tracking database\ALLSTIMULI\i05june05_static_street_boston_p1010885.jpeg';
featureNum = 1;
features = 1:7;
remainFeatures = features;
remainFeatures(featureNum) = [];
saliencyMap1 = saliency(image,featureNum);
saliencyMap2 = saliency(image,remainFeatures);

set(gcf,'Color','w')
subplot(1,3,1);
imshow(imread(image));
title("Original Image",'Interpreter','latex');
axis square

subplot(1,3,2);
imagesc(saliencyMap1);
title("SaliencyMap (feature " + num2str(featureNum) + " present)",'Interpreter','latex');
colormap('gray');
axis square

subplot(1,3,3);
imagesc(saliencyMap2);
title("SaliencyMap (feature " + num2str(featureNum) + " not present)",'Interpreter','latex');
colormap('gray');
axis square

%%
clear all
close all 
clc
addpath(genpath('E:\neuro\Eye tracking database'))
stimfolder = 'Eye tracking database\Eye tracking database\Eye tracking database\ALLSTIMULI';

subjects = {'CNG', 'ajs', 'emb', 'ems', 'ff', 'hp', 'jcw', 'jw', 'kae', 'krl', 'po', 'tmj', 'tu', 'ya', 'zb'};
nImages = 1003;
nFeatures = 7;
nSubjects = randperm(15,5);

score1 = zeros(nFeatures,length(nSubjects)*nImages);
score2 = zeros(nFeatures,length(nSubjects)*nImages);

for featureNum = 7 : nFeatures + 1
    cnt = 1;
    Precentage_of_Process = featureNum/nFeatures
    for i = 1 : nImages
        for j = nSubjects
            datafolder = ['Eye tracking database\Eye tracking database\Eye tracking database\DATA\',subjects{j}];
            [xEye,yEye,xLimit,yLimit] = eyeData(datafolder,stimfolder,i);
            xEye1 = xEye(1:floor(end/2));
            xEye2 = xEye(1+floor(end/2):end);
            yEye1 = yEye(1:floor(end/2));
            yEye2 = yEye(1+floor(end/2):end);
            [xEye1,yEye1] = eyeDataPreprocess(xEye1,yEye1,xLimit,yLimit); 
            [xEye2,yEye2] = eyeDataPreprocess(xEye2,yEye2,xLimit,yLimit);
    
            %Get names of stimuli files in Data Folder:
            files = dir(fullfile(datafolder,'*.mat'));
            [filenames{1:size(files,1)}] = deal(files.name);
            
            load(fullfile(datafolder,filenames{i}));
            stimFile = eval([filenames{i}(1:end-4)]);
            
            imgName = stimFile.imgName;
            if exist(fullfile(stimfolder,imgName))
                image = fullfile(stimfolder,imgName);
            else
                output = 'image file not found';
            end
            try
                if featureNum ~= 8 
                    saliencyMap = saliency(image,featureNum);
                elseif featureNum == 8
                    saliencyMap = saliency(image,1:7);
                end
            catch
                error = 1;
            end
            score1(featureNum,cnt) = rocScoreSaliencyVsFixations(saliencyMap,xEye1,yEye1,[xLimit,yLimit]);
            score2(featureNum,cnt) = rocScoreSaliencyVsFixations(saliencyMap,xEye2,yEye2,[xLimit,yLimit]);
            cnt = cnt + 1;
        end
    end
end
%%
set(gcf,'Color','w')
for i = 1 : 14
    if i < 8
        subplot(2,7,i)
        histogram(score1(i,:),'Normalization','probability')
        hold on
        histogram(score1(8,:),'Normalization','probability')
        xlabel("ROC Score",'interpreter','latex')
        ylabel("Count",'interpreter','latex')
        title("Histogram of ROC Scores For The First 1.5s",'interpreter','latex')
        legend('Feature 1','Features 1 to 7','interpreter','latex')
    else
        subplot(2,7,i)
        histogram(score2(i,:),'Normalization','probability')
        hold on
        histogram(score2(8,:),'Normalization','probability')
        xlabel("ROC Score",'interpreter','latex')
        ylabel("Count",'interpreter','latex')
        title("Histogram of ROC Scores For The Second 1.5s",'interpreter','latex')
        legend('Feature 1','Features 1 to 7','interpreter','latex')
    end
end

%%
set(gcf,'Color','w')
averageScores1 = mean(score1,2,"omitnan");
averageScores1(8) = [];
averageScores1(6) = mean(score1(6,1:3750),2,"omitnan");
averageScores2 = mean(score2,2,"omitnan");
averageScores2(6) = mean(score2(6,1:3750),2,"omitnan");
averageScores2(8) = [];
plot(averageScores1,'Color','r')
hold on
p1 = plot(averageScores1,'*','Color','r');
hold on
plot(averageScores2,'Color','b')
hold on
p2 = plot(averageScores2,'*','Color','b');
p1.HandleVisibility = 'off';
p2.HandleVisibility = 'off';
xlabel("Features",'interpreter','latex')
ylabel("Average ROC Score",'interpreter','latex')
title("Comaparing the average ROC scores of all subjects and images ",'interpreter','latex')
legend('bottom-up','top-down','interpreter','latex')


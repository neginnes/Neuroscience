%%
clear
clc
%compile mex files
addpath(genpath('E:\neuro\WinSparseNet\Code\Matlab\Bruno\sparsenet'))
mex -v cgf.c nrf/brent.c nrf/frprmn.c nrf/linmin.c nrf/mnbrak.c nrf/nrutil.c -Inrf

%% Part1
clear all
close all
clc
load('IMAGES.mat');
VAR_GOAL=0.1;
for i = 1 : 10
    IMAGES(:,:,i) = IMAGES(:,:,i)./sqrt(var(IMAGES(:,:,i),0,'all')) * sqrt(VAR_GOAL);
end
A = rand(16*16,192)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));
sparsenet

%% Part2.1
clear all
close all
clc
M = 10;
N = 150;
VAR_GOAL=0.1;
IMAGES = [];
directory = dir('CroppedYale\CroppedYale');
randIm = 3 + randperm(size(directory,1)-3,10);
cnt = 1;
for k = randIm
    D = dir([directory(k).folder '\' directory(k).name]);
    Rnd = randi([5,size(D, 1)]);
    while(contains([D(Rnd).folder '\' D(Rnd).name],"bad"))
        Rnd = randi([5,size(D, 1)]);
    end
    img = imread([D(Rnd).folder '\' D(Rnd).name]);
    img = imresize(img,[N,N]);
    IMAGES(:,:,cnt) = double(img);
    cnt = cnt + 1;
end
[fx, fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
rho=sqrt(fx.*fx+fy.*fy);
f_0=0.4*N;
filt=rho.*exp(-(rho/f_0).^4);
for i = 1:M
    IMAGES(:,:,i) = IMAGES(:,:,i) - mean(IMAGES(:,:,i),"all");
    IMAGES(:,:,i) = zscore(IMAGES(:,:,i));
    If=fft2(IMAGES(:,:,i));
    imagew=real(ifft2(If.*fftshift(filt)));
    IMAGESVec(:,i)=reshape(imagew,N^2,1);
end
IMAGESVec=sqrt(0.1)*IMAGESVec/sqrt(mean(var(IMAGESVec)));
IMAGES = reshape(IMAGESVec,N,N,M);

for i=1:M
    IMAGES(:,:,i) = IMAGES(:,:,i)/sqrt(var(IMAGES(:,:,i),0,"all"))*sqrt(VAR_GOAL);
end

A = rand(12*12,100)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));
sparsenet


%% Part2.2
clear all
close all
clc
load('mnist.mat')
IMAGES = test.images;
M = 10;
N = size(IMAGES,1);
VAR_GOAL=0.1;
randIm = randperm(size(IMAGES,3),10);

[fx, fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
rho=sqrt(fx.*fx+fy.*fy);
f_0=0.4*N;
filt=rho.*exp(-(rho/f_0).^4);
for i = 1:M
    IMAGES(:,:,i) = IMAGES(:,:,i) - mean(IMAGES(:,:,i),"all");
    IMAGES(:,:,i) = zscore(IMAGES(:,:,i));
    If=fft2(IMAGES(:,:,i));
    imagew=real(ifft2(If.*fftshift(filt)));
    IMAGESVec(:,i)=reshape(imagew,N^2,1);
end
IMAGESVec=sqrt(0.1)*IMAGESVec/sqrt(mean(var(IMAGESVec)));
IMAGES = reshape(IMAGESVec,N,N,M);

for i=1:M
    IMAGES(:,:,i) = IMAGES(:,:,i)/sqrt(var(IMAGES(:,:,i),0,"all"))*sqrt(VAR_GOAL);
end

A = rand(6*6,100)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));
sparsenet



%% Part2.3
clear all
close all
clc
M = 10;
N = 150;
VAR_GOAL=0.1;
IMAGES = [];
directory = dir('Homework2-Caltech101-master\Homework2-Caltech101-master\101_ObjectCategories');
randIm = 3 + randperm(size(directory,1)-3,10);
cnt = 1;
for k = randIm
    D = dir([directory(k).folder '\' directory(k).name]);
    Rnd = randi([3,size(D,1)]);
    while(contains([D(Rnd).folder '\' D(Rnd).name],"bad"))
        Rnd = randi([5,size(D, 1)]);
    end
    img = im2gray(imread([D(Rnd).folder '\' D(Rnd).name]));
    img = imresize(img,[N,N]);
    IMAGES(:,:,cnt) = double(img);
    cnt = cnt + 1;
end
[fx, fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
rho=sqrt(fx.*fx+fy.*fy);
f_0=0.4*N;
filt=rho.*exp(-(rho/f_0).^4);
for i = 1:M
    IMAGES(:,:,i) = IMAGES(:,:,i) - mean(IMAGES(:,:,i),"all");
    IMAGES(:,:,i) = zscore(IMAGES(:,:,i));
    If=fft2(IMAGES(:,:,i));
    imagew=real(ifft2(If.*fftshift(filt)));
    IMAGESVec(:,i)=reshape(imagew,N^2,1);
end
IMAGESVec=sqrt(0.1)*IMAGESVec/sqrt(mean(var(IMAGESVec)));
IMAGES = reshape(IMAGESVec,N,N,M);

for i=1:M
    IMAGES(:,:,i) = IMAGES(:,:,i)/sqrt(var(IMAGES(:,:,i),0,"all"))*sqrt(VAR_GOAL);
end

A = rand(4*4,100)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));
sparsenet


%% Part3
clear all
close all
clc
M = 10;
N = 150;
VAR_GOAL=0.1;
IMAGES = [];
vidObj = VideoReader('Data\BIRD.avi');
for i = 1:M
    img = readFrame(vidObj);
    img = im2gray(img);
    img = imresize(img,[N,N]);
    IMAGES(:,:,i) = img;
end
[fx, fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
rho=sqrt(fx.*fx+fy.*fy);
f_0=0.4*N;
filt=rho.*exp(-(rho/f_0).^4);
for i = 1:M
    IMAGES(:,:,i) = IMAGES(:,:,i) - mean(IMAGES(:,:,i),"all");
    IMAGES(:,:,i) = zscore(IMAGES(:,:,i));
    If=fft2(IMAGES(:,:,i));
    imagew=real(ifft2(If.*fftshift(filt)));
    IMAGESVec(:,i)=reshape(imagew,N^2,1);
end
IMAGESVec=sqrt(0.1)*IMAGESVec/sqrt(mean(var(IMAGESVec)));
IMAGES = reshape(IMAGESVec,N,N,M);

for i=1:M
    IMAGES(:,:,i) = IMAGES(:,:,i)/sqrt(var(IMAGES(:,:,i),0,"all"))*sqrt(VAR_GOAL);
end

A = rand(8*8,100)-0.5;
A = A*diag(1./sqrt(sum(A.*A)));
sparsenet


%%  
M = 10;
N = 288;
VAR_GOAL=0.1;
[fx, fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
rho=sqrt(fx.*fx+fy.*fy);
f_0=0.4*N;
filt=rho.*exp(-(rho/f_0).^4);
nPatches = N/8 * N/8;
noise_var= 0.01;
beta= 2.2;
sigma=0.316;
tol=.01;
Coefs =[];

vidObj = VideoReader('Data\BIRD.avi');
for i = 1:2*M
    if(i<=M)
        continue
    end
    img = readFrame(vidObj);
    img = im2gray(img);
    img = imresize(img,[N,N]);
    img = double(im2gray(img));
    img = img - mean(img,"all");
    img = zscore(img);
    If=fft2(img);
    imagew=real(ifft2(If.*fftshift(filt)));
    IMAGESVec=reshape(imagew,N^2,1);
    IMAGESVec=sqrt(0.1)*IMAGESVec/sqrt(mean(var(IMAGESVec)));
    img = reshape(IMAGESVec,N,N);
    img = img/sqrt(var(img,0,"all"))*sqrt(VAR_GOAL);
    W = N / sqrt(nPatches);
    cnt = 1;
    Patches =[];
    for p = 1 : W : N-(W-1)
        for q = 1 : W : N-(W-1)
            Patches(:,:,cnt) = img(p:p+W-1,q:q+W-1);
            cnt = cnt + 1;
        end
    end
    Patches = reshape(Patches,size(Patches,1)*size(Patches,2),size(Patches,3));
    % calculate coefficients for these data via conjugate gradient routine 
    S=cgf_fitS(A,Patches,noise_var,beta,sigma,tol);
    Coefs(:,:,i-M) = S;
end 

%%

writerObj = VideoWriter("Part3");
writerObj.FrameRate = 20;
open(writerObj);
fig = figure();
set(gcf,'Color','w')
for i = 1 : size(Coefs,3)
    subplot(2,2,1)
    imagesc(zscore(Coefs(:,:,i)))
    colormap('gray')
    colorbar()
    title(['Normalized Coefficients, Frame = ',num2str(10+i)],'interpreter','latex')
    

    subplot(2,2,2)
    [m1,m2] = find(Coefs(:,:,i)==max(Coefs(:,:,i),[],"all"));
    scatter(m2,m1,'filled','blue')
    hold on
    [m1,m2] = find(Coefs(:,:,i)==min(Coefs(:,:,i),[],"all"));
    scatter(m2,m1,'filled','red')
    axis('square')
    legend('Max','Min','interpreter','latex')
    title(['Min and Max, Frame = ',num2str(10+i)],'interpreter','latex')
    xlim([0,size(Coefs,2)])
    ylim([0,size(Coefs,1)])
    hold off

   
    subplot(2,2,3)
    histogram(Coefs(:,:,i),'Normalization','probability')
    title(['Histogram of Coefficients, Frame = ',num2str(10+i)],'interpreter','latex')
    axis('square')

    subplot(2,2,4)
    scatter(i,max(Coefs(:,:,i),[],"all"),'blue')
    hold on
    scatter(i,min(Coefs(:,:,i),[],"all"),'red')
    hold on
    scatter(i,mean(Coefs(:,:,i),"all"),'black')
    hold on
    scatter(i,var(Coefs(:,:,i),0,"all"),'green')
    legend('Max','Min','Mean','Var','interpreter','latex')
    title(['Statistics Through Time, Frame = ',num2str(10+i)],'interpreter','latex')
    axis('square')

    pause(1)
    hold on
    for j = 1:10
        frame = getframe(fig);
        writeVideo(writerObj,frame);
    end
end

close(writerObj)

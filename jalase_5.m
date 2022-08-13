%%%%%MOHSEN ABYARI RIZI
%%%BCI test 
%%%5 sieson

clc
clear all
close all

%% filter design for Alpha BAND EXTARCTION (8-13HZ)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fs = 128;  % Sampling Frequency

Fstop1 = 7;    % First Stopband Frequency
Fpass1 = 8;    % First Passband Frequency
Fpass2 = 12;   % Second Passband Frequency
Fstop2 = 13;   % Second Stopband Frequency
Dstop1 = 0.1;  % First Stopband Attenuation
Dpass  = 0.1;  % Passband Ripple
Dstop2 = 0.1;  % Second Stopband Attenuation
dens   = 20;   % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                          0], [Dstop1 Dpass Dstop2]);

% Calculate the coefficients using the FIRPM function.
b  = firpm(N, Fo, Ao, W, {dens});
Hd1 = dfilt.dffir(b);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% filter design for Betaa BAND EXTARCTION (13-25HZ)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fs = 128;  % Sampling Frequency

Fstop1 = 12;    % First Stopband Frequency
Fpass1 = 13;    % First Passband Frequency
Fpass2 = 24;   % Second Passband Frequency
Fstop2 = 25;   % Second Stopband Frequency
Dstop1 = 0.1;  % First Stopband Attenuation
Dpass  = 0.1;  % Passband Ripple
Dstop2 = 0.1;  % Second Stopband Attenuation
dens   = 20;   % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                          0], [Dstop1 Dpass Dstop2]);

% Calculate the coefficients using the FIRPM function.
b  = firpm(N, Fo, Ao, W, {dens});
Hd2 = dfilt.dffir(b);


load 'dataset_BCIcomp1.mat'
for i=1:140
    data(:,:)=x_train(:,:,140);
    c3=data(:,1);
    cz=data(:,2);
    c4=data(:,3);
    [Y1,T1]=phasespace(c3,2,3);
    [Y2,T2]=phasespace(c4,2,3);
    %%phase space recontraction C3
     r1=Y1(:,1);
     r2=Y1(:,2);
    %%phase space recontraction C3
     r3=Y2(:,1);
     r4=Y2(:,2);

%%%%alpha band
     s1=filter(Hd1,r1);
     s2=filter(Hd1,r2);
     s3=filter(Hd1,r3);
     s4=filter(Hd1,r4);
%%%%beta band
     s5=filter(Hd2,r1);
     s6=filter(Hd2,r2);
     s7=filter(Hd2,r3);
     s8=filter(Hd2,r4);

         f1=max(abs(fft(s1)));
         f2=max(abs(fft(s2)));
         f3=max(abs(fft(s3)));
         f4=max(abs(fft(s4)));
         f5=max(abs(fft(s5)));
         f6=max(abs(fft(s6)));
         f7=max(abs(fft(s7)));
         f8=max(abs(fft(s8)));

       feature(i,:)=[f1 f2 f3 f4 f5 f6 f7 f8];
    
end
    input=feature;
    output=y_train-1;

x = input';
t = output';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 3;
net = patternnet(hiddenLayerSize, trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 5/100;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
figure, plotconfusion(t,y)
%figure, plotroc(t,y)

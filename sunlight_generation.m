clear all; close all; clc
%% load data
[a b c] = xlsread('sunlight.xlsx');
P=a(:,4:18); %input
T=a(:,19); %output
%% normalize data 
minp = min(P)'; maxp = max(P)';
mint = min(T); maxt = max(T);
for j = 1:size(P,2)
Normal_Input(:,j) = -1+(P(:,j)-minp(j))./(maxp(j)-minp(j))*2;
end
Normal_Output = -1+(T-mint)./(maxt-mint)*2;
pn=Normal_Input';
tn=Normal_Output';

pn_train = pn(:, 1:48);
tn_train = tn(:, 1:48);
pn_valid = pn(:, 49:60);
tn_valid = tn(:, 49:60);
%% apply ANN
%http://kr.mathworks.com/help/nnet/ref/traingd.html 
net = newff(minmax(pn),[6 6 1],{'tansig','tansig','tansig'},'traingdm');
net.trainParam.epochs = 2000; %iteration number
net=train(net,pn_train,tn_train)
Output=sim(net,pn_train);
%% denormalize data
tr_sim=postmnmx(Output,mint,maxt);
tr_obs=postmnmx(tn_train,mint,maxt);
%% Apply ANN to validation set
Output_valid = sim(net, pn_valid);

%% Denormalize data for validation set
va_sim = postmnmx(Output_valid, mint, maxt);
va_obs = postmnmx(tn_valid, mint, maxt);
%% Calculate RMSE for validation set
r2_va=corr(va_obs', va_sim')^2;
%% calculate R square
r2_tr=corr(tr_obs',tr_sim')^2;

%% plot
subplot(2,2,1)
h=plot(tr_obs','ok');
hold on
h2=plot(tr_sim','-r');
hold on
legend([h h2],'Observation','Simulation')
xlim([1 50])
subplot(2,2,3)
h=plot(va_obs','ok')
hold on
h2=plot(va_sim','-r')
hold on
legend([h h2],'observation','Simulation')
xlim([1 30])
subplot(2,2,2)
scatter(tr_obs,tr_sim)
title(['R2=' num2str(r2_tr)],'fontsize', 15, 'fontname','Times')
subplot(2,2,4)
scatter(va_obs',va_sim')
title(['R2=' num2str(r2_va)],'fontsize', 15, 'fontname','Times')
%% plot
x = [1, 2];
y = [r2_tr, r2_va];
figure;
bar(x,y);
title('model performance');
set(gca,'xticklabel', {'train', 'validation'});





clc;
clear;
% set path
addpath('.\data');
addpath('.\util');
% load dataset
load('outdoor_scene.mat');
warning('off');
FeaNumCandi = [100:100:500]; % feature dimension
[~,class_num] = size(tag);  
[num, diemnsion] = size(D); 
k = class_num;
label = zeros(num, 1);
for i = 1:num
   label(i)=find(tag(i,:)==1); 
end
%% run ACSL
[P, F, S, W] = ACSL(X, 1, 2e-2, 1e-2,  max(label), length(X.data));
%% calculate weight
W1 = [];
for k = 1:diemnsion
    W1 = [W1 norm(P(k,:),2)];
end
%% test stage
[~,index] = sort(W1,'descend');
for j = 1:length(FeaNumCandi)
    new_fea = X.fea(:,index(1:FeaNumCandi(j)));
    idx = kmeans(new_fea, class_num);
    res = bestMap(label,idx);
    RS_data{1}.AC(1,j)= length(find(label == res))/length(label); % calculate ACC 
    RS_data{1}.NMI(1,j) = MutualInfo(label,idx); % calculate NMI
    disp(['ACSL ','Selected feature num: ',num2str(FeaNumCandi(j)),', Clustering MIhat: ',num2str(RS_data{1}.NMI(1,j)), ', AC: ',num2str(RS_data{1}.AC(1,j))]);
end

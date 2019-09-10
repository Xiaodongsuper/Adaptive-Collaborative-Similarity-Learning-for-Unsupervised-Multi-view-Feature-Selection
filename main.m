clc;
clear;
addpath('.\data');
addpath('.\util');
load('outdoor_scene.mat');
warning('off');
FeaNumCandi = [100:100:500];
[~,class_num] = size(tag);
[num, diemnsion] = size(D);
k = class_num;
label = zeros(num, 1);
for i = 1:num
        label(i)=find(tag(i,:)==1);
end
[P, F, S, W] = ACSL(X,1e-2,2e-2,1e-2, 4,  max(label), length(X.data));
W1 = [];
for k = 1:diemnsion
    W1 = [W1 norm(P(k,:),2)];
end
[~,index] = sort(W1,'descend');
for j = 1:length(FeaNumCandi)
    new_fea = X.fea(:,index(1:FeaNumCandi(j)));
    idx = kmeans(new_fea, class_num);
    res = bestMap(label,idx);
    RS_data{i}.AC(7,j)= length(find(label == res))/length(label);
    RS_data{i}.NMI(7,j) = MutualInfo(label,idx);
  disp(['ACSL ','Selected feature num: ',num2str(FeaNumCandi(j)),', Clustering MIhat: ',num2str(RS_data{i}.NMI(7,j)), ', AC: ',num2str(  RS_data{i}.AC(7,j))]);
end
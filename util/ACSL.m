function [P, F, S, W] = ACSL(X, alpha, beta, gamma, class_num, view_num)
    %% input: dataset X
    %       parameters alpha and beta
    %       parameter gamma for updateing P
    %       the number of class class_num
    %       the number of view view_num
    %  output:
    %       projective matrix P
    %       cluster indicator matrix F
    %       global similarity S
    %       view weight W
    % author:Xiao Dong 2018.1.17
    %% initial parameter variables
    % initial distance parameters of weight_function
    options = [];
    options.Metric = 'HeatKernel';
    options.NeighborMode = 'KNN';
    options.WeightMode = 'Cosine';
    options.k = 5;
    [n, d] = size(X.fea);
    k = class_num;
    P = rand(d, k);       
    F = rand(n, k);
    W = ones(view_num, n)*(1/view_num);
    S = zeros(n, n);
    epsil = 0.01;   % convergence threshold
    num = 0;  %iteration number flag
    Gamma = diag(ones(d,1));
    eps = 0.0001;
    %% initial similarity matrix for each view
    for v = 1:view_num
         S_d.data{v} = constructW(X.data{v},options);
         for i =1:n
            S(:, i) = S(:, i) + W(v,i)*S_d.data{v}(:,i); 
         end
    end 
    S = S./repmat(sqrt(sum(S.^2,1)),size(S,1),1);
    
   %% according to algorithm 1 initial F
    Q = X.fea'*X.fea + gamma*Gamma;
    L_s = diag(sum(S,2))-S;
    Mat_F = L_s + beta*eye(n, n) - beta*(X.fea/Q*X.fea');
    [M, lambda] = eig(Mat_F);
    lambda(isnan(lambda)) = 0;
    [lambda, ind] = sort(diag(lambda), 'ascend');
    F = M(:,ind(1:k)); 
    %% optimization 
    while 1        
       %% calculate objective function value1
        temp_formulation1 = 0;
        for i =1:n
            temp_S_j = zeros(n,1);
            for v = 1:view_num
                temp_S_i = temp_S_j + W(v,i)*S_d.data{v}(:,i);
            end
            temp_formulation1 = temp_formulation1 + norm(S(:,i)-temp_S_i,'fro')^2;
        end
        temp_formulation2 = alpha*trace(F'*(diag(sum(S,2))-S)*F);        
        temp_formulation31 = norm(X.fea*P-F,'fro')^2;
        temp_formulation32 = 0;
        for i = 1:d
            temp_formulation32 = temp_formulation32 + norm(P(i,:),2);
        end
        temp_formulation3 = beta*(temp_formulation31+ gamma*temp_formulation32);
        value1 = temp_formulation1 + temp_formulation2 + temp_formulation3;
        
        %% update P
        while 1
            value_P1 = norm(X.fea*P - F,'fro') + gamma*trace(P'*Gamma*P);
            temp_gamma = zeros(d,1);
            for i = 1:d
                temp_gamma(i) = 1/(2*sqrt(P(i,:)*P(i,:)')+epsil);
            end
            Gamma  = diag(temp_gamma);
            P = (X.fea'*X.fea + gamma*Gamma)\X.fea'*F;
            value_P2 = norm(X.fea*P - F,'fro') + gamma*trace(P'*Gamma*P);
            if abs(value_P1-value_P2)<eps
                break;
            end
        end      
        %% update F
        Q = X.fea'*X.fea + gamma*Gamma;
        L_s = diag(sum(S,2))-S;
        Mat_F = L_s + beta*eye(n, n) - beta*(X.fea/Q*X.fea');
        [M, lambda] = eig(Mat_F);
        % Sort eigenvalues and eigenvectors in descending order
        lambda(isnan(lambda)) = 0;
        [lambda, ind] = sort(diag(lambda), 'ascend');
        F = M(:,ind(1:k));       
        %% update S
        for j = 1:n
            A_j = zeros(n, 1);
            for i =1:n
                A_j(i) = norm(F(j,:)-F(i,:),2)^2;
            end
            formulation_part = zeros(n,1);
            for i = 1:view_num
                formulation_part = formulation_part + W(i,j)*S_d.data{v}(:,j);
            end
            S(:,j) = EProjSimplex_new(formulation_part-(alpha/2)*A_j,k);
        end       
        %% update W
        for j = 1:n
            B_j = zeros(n,view_num);
            for v = 1:view_num
                B_j(:,v) = S(:,j)-S_d.data{v}(:,j);
            end
            W(:,j) = ((B_j'*B_j+eye(v)*1e-5)\ones(view_num,1))/(ones(view_num,1)'/(B_j'*B_j+eye(v)*1e-5)*ones(view_num,1));
        end
       %% calculate objective function value2
        temp_formulation1 = 0;
        for i =1:n
            temp_S_j = zeros(n,1);
            for v = 1:view_num
                temp_S_i = temp_S_j + W(v,i)*S_d.data{v}(:,i);
            end
            temp_formulation1 = temp_formulation1 + norm(S(:,i)-temp_S_i,'fro')^2;
        end
        temp_formulation2 = alpha*trace(F'*(diag(sum(S,2))-S)*F);
        
        temp_formulation31 = norm(X.fea*P-F,'fro')^2;
        temp_formulation32 = 0;
        for i = 1:d
            temp_formulation32 = temp_formulation32 + norm(P(i,:),2);
        end
        temp_formulation3 = beta*(temp_formulation31+ gamma*temp_formulation32);
        value2 = temp_formulation1 + temp_formulation2 + temp_formulation3;
        
        num = num + 1;
        %% convergence condition
        if(abs(value1-value2)<epsil)
            break;
        end       
    end
end

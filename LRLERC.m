function [D, V, A, B] = LRLERC(X,W,InitD, InitV, ProK,cluster_n, Pgamma,Plamda,maxStep,conCriterion)
%% Low-rank linear embedding for robust clustering
% ||XAB-DVAB||_2,1+ r sum_ij(||xi-xjAB||_2,1*wij)+lamda*||AB||_2,1
% X:数据集， n*m，行向量形式
% W:由LLE导出的相似度矩阵
% ProK：选择的特征数目
%Pgamma，Plamda：参数
% maxStep：最大迭代步数
% conCriterion ：收敛条件
%A: m*ProK;
%B: ProK*m  
%D: n*cluster_n  行向量
%V: cluster_n*m
%InitD:  n*cluster_n 
%initV: cluster_n*m

%% --------------------begin------------------------------------------------
data_n = size(X,1);
dim_n = size(X,2);
 
D = InitD;
V = InitV;

A = eye(dim_n,ProK);    %初始化为单位阵
B = eye(ProK, dim_n);

steps=0;
converged=false;

while ~converged && steps<=maxStep
     
    steps=steps+1;      
    V_Old = V;
    
     %%%%%%%%%%%%%%%%%%%%%（求B）      
     tempG = X*A*B - D*V*A*B;
     tempG21 = sqrt( sum( tempG.^2, 2 ) );
     G = diag( 1./ ( 2 .* tempG21 + eps ) );
     
     E = X'*G*X - (V'*D'*G*X + X'*G*D*V) + V'*D'*G*D*V;
     
     dis_X_XAB = EuDist2(X, X*A*B);  
     Wbar = W./( 2.* dis_X_XAB + eps );            
     L = diag( sum(Wbar,2) );
     
     tempQ = A*B;
     tempQ21 = sqrt( sum( tempQ.^2, 2 ) );
     Q = diag( 1./ ( 2 .* tempQ21 + eps ) );
          
     TE = E + Pgamma*X'*L*X + Plamda*Q;
     
     B = Pgamma* pinv(A'* (TE) *A) * A'*X'*Wbar*X;     
    
     %%%%%%%%%%%%%%%%%%%%%%%%%%%（求A ）
     
     AA = pinv( TE ) * X'*Wbar*X*X'*Wbar'*X;
     [eigvector, eigvalue] = eig(AA);

     eigvalue = diag(eigvalue);            
     [junk, index] = sort(eigvalue,'descend');      
     eigvalue = eigvalue(index);
     eigvector = eigvector(:, index);
    
%     maxEigValue = max(abs(eigvalue));
%     eigIdx = find(abs(eigvalue)/maxEigValue < 1e-12);
%     eigvalue (eigIdx) = [];
%     eigvector (:,eigIdx) = [];
    
     if ProK < length(eigvalue)             %取最大的k个特征值对应的特征向量
       eigvalue = eigvalue(1:ProK);
       eigvector = eigvector(:, 1:ProK);
     end
     A = eigvector;
    
     nsmp=size(A,2);   %dim_n*ProK  行向量
     for i=1:nsmp
       A(:,i)=A(:,i)/norm(A(:,i),2);
     end                                         

    %%%%%%%%%%%%%%%%%%%%%%%%%%（求D ）
     dis_XA_VA = EuDist2(X*A,V*A);   %%距离，已经开方，L21
     
     TempDis = dis_XA_VA';
     [minV, minIdx] = min( TempDis );

     D = zeros( data_n, cluster_n);
     for i=1:1:data_n
         D(i, minIdx(i)) = 1;
     end
     
     %%%%%%%%%%%%%%%%%%%%%%%%%%（求V ）
     V = pinv(D'*G*D) * D'*G*X;
  
    criterion = 0;
    for i=1:1:cluster_n
        tempV_Vold = norm(V(i,:)-V_Old(i,:),2);
        if criterion < tempV_Vold
            criterion = tempV_Vold;
        end
    end
    if criterion < conCriterion
        converged=true;
    end       
end 
%%--------------------end--------------------------------------------------
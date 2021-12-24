    clc;
    clear;   

    filepath = 'Iris.txt';
    data = textread(filepath);   
    filepath_gt = 'gnd.txt'; 
    gnd = textread(filepath_gt);

    X = data;     
    Xorgdim = X;
   
%       [eigvector, eigvalue, elapse] = PCA_dencai(X, 60);%input row vector is a sample image, for pre-dimensionality reduction to accelerate the speed 
%       projection=eigvector;%(:,1:dim);
%       Xorgdim = X;
%       X = X* projection;
   
    data_n = size(X,1);
    dim_n = size(X,2);   
   
 %   pro_dim_n = min( min(data_n-1, dim_n), 60);   
    
    cluster_n = length(unique(gnd));          %类数  
    maxStep = 100;            %最大迭代次数
    conCriterion = 0.001;     %迭代终止条件
 
    %------- construciton method 
     options = [];
     options.k = 10;
     options.NeighborMode = 'KNN';
     W = Wconstruct_NPE(options, X);
     W = BuildAdjacency(W);  

     %% %% ----------Low-rank linear embedding for robust clustering

     InitD=zeros( data_n, cluster_n);

      %-----------random initialization ------------
%     initU = rand( cluster_n , data_n ); %初始化隶属度矩阵
%     suminitU = sum(initU);                  %归一化
%     suminitU = ones(cluster_n,1)*suminitU;
%     initU = initU ./ suminitU;
%     
%     [maxInitU, maxIdx] = max(initU);
%     for i=1:1:data_n;
%         InitD(i, maxIdx(i)) = 1;
%     end

    % -------Initialization with k-means -------------
     %rng(1); % For reproducibility
     [DKmeans,VKmeans] = kmeans(X,cluster_n);

    for i=1:1:data_n;
        InitD(i, DKmeans(i)) = 1;
    end
    InitV = VKmeans; 

    GammaRegion = [10^-4,10^-3,10^-2,10^-1,10^0,10^1,10^2];
    LamdaRegion = [10^-3,10^-2,10^-1,10^0,10^1,10^2,10^3];
    
    ProKLrlerc = dim_n;
    PgammaLrlerc = 1;
    PlamdaLrlerc = 1;
    ACCindexLrlerc = 0;
    NMIindexLrlerc = 0;
    ComResultLrlerc = [];
    steps = 0;
    
    Circ = 1;
    FAccLrlerc = 0;
    for ProK = 2:1:dim_n
    %for ProK=5:5:pro_dim_n
        for gi=1:1:length(GammaRegion)
            Pgamma = GammaRegion( gi );
            for mi=1:1:length(LamdaRegion);
                Plamda = LamdaRegion( mi );
                [D, V, A, B] = LRLERC(X,W,InitD,InitV, ProK,cluster_n, Pgamma,Plamda,maxStep,conCriterion);
                
                FDLrlerc = D';
                [TempMax, grpsLrlerc]=max(FDLrlerc);
                grpsLrlerc = grpsLrlerc(1:length(gnd));
                ACCindexTemp= ACC2(gnd, grpsLrlerc, cluster_n);
                NMIindexTemp = NMI(gnd, grpsLrlerc);
             
                ComResultLrlerc(Circ,:) = [ProK, Pgamma, Plamda, ACCindexTemp, NMIindexTemp];                
                
                if FAccLrlerc< ACCindexTemp
                   ACCindexLrlerc  = ACCindexTemp;
                   NMIindexLrlerc  = NMIindexTemp;
                   FAccLrlerc  = ACCindexTemp;
                   ProKLrlerc  = ProK;
                   PgammaLrlerc  = Pgamma;
                   PlamdaLrlerc  = Plamda;               
                end
                Circ = Circ + 1;
                fprintf('ID: %f, ProK = %f, PGamma = %f, PLamda = %f, ACC = %f, NMI = %f\n', Circ, ProK, Pgamma, Plamda,ACCindexTemp,NMIindexTemp);                
            end
        end
    end
    fprintf('RCLR Clustering: ProK = %f, PGamma = %f, PLamda = %f, ACC = %f, NMI = %f\n',ProKLrlerc, PgammaLrlerc, PlamdaLrlerc,ACCindexLrlerc,NMIindexLrlerc );   
    

     
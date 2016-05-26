%% initialize data

setup() ;
% setup('useGpu', true);
imdb = load('data/text_imdb.mat') ; %%%%%%%%%%%%%Change to fit data


%% initialize net

% set parameter

opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
%opts.weightInitMethod = 'xavierimproved' ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'alexnet' ;
opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
%%%opts = vl_argparse(opts, varargin) ;

net = initializeAlexnet(opts);
%%%net = cnn_imagenet_init(alexnet) ;

% Display network
vl_simplenn_display(net) ;

%% Train
trainOpts.expDir = 'data/alex_net' ;
trainOpts.gpus = [] ;
% Uncomment for GPU training:
%trainOpts.expDir = 'data/alex_net-gpu' ;
%trainOpts.gpus = [1] ;
trainOpts.batchSize = 128 ;
trainOpts.learningRate = 0.001 ;
trainOpts.plotDiagnostics = false ;
%trainOpts.plotDiagnostics = true ; % Uncomment to plot diagnostics
trainOpts.numEpochs = 25 ;
trainOpts.errorFunction = 'none' ;

net = cnn_train(net, imdb, @getBatch, trainOpts) ;   %%%%%%%%Change to fit data!

% Deploy: remove loss
net.layers(end) = [] ;

%% evaluate


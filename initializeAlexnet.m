function net = initializeAlexnet(opts)
%INITIALIZELARGECNN  Initialize an Alexnet
%   NET = INITIALIZELARGECNN() returns the SimpleNN model NET.
net.meta.normalization.imageSize = [227, 227, 3] ;
net = alexnet(net, opts) ;
bs = 256 ;

% final touches
switch lower(opts.weightInitMethod)
  case {'xavier', 'xavierimproved'}
    net.layers{end}.weights{1} = net.layers{end}.weights{1} / 10 ;
end
net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;

% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad, init_bias)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                           'weights', {{init_weight(opts, h, w, in, out, 'single'), zeros(out, 1, 'single')}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0], ...
                           'opts', {convOpts}) ;
if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
                             'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single'), zeros(out, 2, 'single')}}, ...
                             'learningRate', [2 1 0.05], ...
                             'weightDecay', [0 0]) ;
end
net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;

% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end

% --------------------------------------------------------------------
function net = add_norm(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'normalize', ...
                             'name', sprintf('norm%s', id), ...
                             'param', [5 1 0.0001/5 0.75]) ;
end

% --------------------------------------------------------------------
function net = add_dropout(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'dropout', ...
                             'name', sprintf('dropout%s', id), ...
                             'rate', 0.5) ;
end

% --------------------------------------------------------------------
function net = alexnet(net, opts)
% --------------------------------------------------------------------

net.layers = {} ;

net = add_block(net, opts, '1', 11, 11, 3, 96, 4, 0) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;


net = add_block(net, opts, '2', 5, 5, 48, 256, 1, 2) ;
net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;


net = add_block(net, opts, '3', 3, 3, 256, 384, 1, 1) ;
net = add_block(net, opts, '4', 3, 3, 192, 384, 1, 1) ;
net = add_block(net, opts, '5', 3, 3, 192, 256, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '6', 6, 6, 256, 4096, 1, 0) ;
net = add_dropout(net, opts, '6') ;

net = add_block(net, opts, '7', 1, 1, 4096, 4096, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, 4096, 10, 1, 0) ;
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end
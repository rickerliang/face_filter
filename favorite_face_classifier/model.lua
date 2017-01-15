require 'torch'
require 'nn'
require 'utils'
local createModel = require 'resnet'
local weightInit = require 'weight-init'

local function buildModel()
    print('building model...')
    local convnet = nn.Sequential()
    -- input plane RGB 3x96x96
    convnet:add(cudnn.SpatialConvolution(3, 16, 4, 4))
    convnet:add(cudnn.SpatialBatchNormalization(16))
    convnet:add(nn.LeakyReLU(0.01))
    convnet:add(cudnn.SpatialMaxPooling(4, 4, 3, 3))
    
    convnet:add(cudnn.SpatialConvolution(16, 32, 4, 4))
    convnet:add(cudnn.SpatialBatchNormalization(32))
    convnet:add(nn.LeakyReLU(0.01))
    convnet:add(cudnn.SpatialMaxPooling(4, 4, 3, 3))
    
    convnet:add(cudnn.SpatialConvolution(32, 32, 4, 4))
    convnet:add(cudnn.SpatialBatchNormalization(32))
    convnet:add(nn.LeakyReLU(0.01))
    convnet:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))

    convnet:add(cudnn.SpatialConvolution(32, 64, 2, 2))
    convnet:add(cudnn.SpatialBatchNormalization(64))
    weightInit(convnet, 'kaiming')

    local classifier = nn.Sequential()
    classifier:add(nn.Reshape(64))
    classifier:add(nn.Linear(64, 256))
    classifier:add(cudnn.BatchNormalization(256))
    classifier:add(nn.LeakyReLU(0.01))
    classifier:add(nn.Linear(256, 2))
    classifier:add(cudnn.LogSoftMax())
    weightInit(classifier, 'kaiming')

    local model = nn.Sequential()
    model:add(convnet)
    model:add(classifier)
    local negWeight = 1.0 / (1.0 + opt.negExampleFactor)
    local loss = nn.ClassNLLCriterion(torch.Tensor{1.0 - negWeight, negWeight})
    print(model)
    return model, loss
end

local function inceptionModel()
   local net = nn.Sequential()

   net:add(nn.SpatialConvolutionMM(3, 64, 7, 7, 2, 2, 3, 3))
   net:add(nn.SpatialBatchNormalization(64))
   net:add(nn.ReLU())

   -- The FaceNet paper just says `norm` and that the models are based
   -- heavily on the inception paper (http://arxiv.org/pdf/1409.4842.pdf),
   -- which uses pooling and normalization in the same way in the early layers.
   --
   -- The Caffe and official versions of this network both use LRN:
   --
   --   + https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
   --   + https://github.com/google/inception/blob/master/inception.ipynb
   --
   -- The Caffe docs at http://caffe.berkeleyvision.org/tutorial/layers.html
   -- define LRN to be across channels.
   net:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
   net:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))

   -- Inception (2)
   net:add(nn.SpatialConvolutionMM(64, 64, 1, 1))
   net:add(nn.SpatialBatchNormalization(64))
   net:add(nn.ReLU())
   net:add(nn.SpatialConvolutionMM(64, 192, 3, 3, 1, 1, 1))
   net:add(nn.SpatialBatchNormalization(192))
   net:add(nn.ReLU())

   net:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))
   net:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))

   -- Inception (3a)
   net:add(nn.Inception{
     inputSize = 192,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {128, 32},
     reduceSize = {96, 16, 32, 64},
     pool = nn.SpatialMaxPooling(3, 3, 1, 1, 1, 1),
     batchNorm = true
   })

   -- Inception (3b)
   net:add(nn.Inception{
     inputSize = 256,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {128, 64},
     reduceSize = {96, 32, 64, 64},
     pool = nn.SpatialLPPooling(256, 2, 3, 3, 1, 1),
     batchNorm = true
   })

   -- Inception (3c)
   net:add(nn.Inception{
     inputSize = 320,
     kernelSize = {3, 5},
     kernelStride = {2, 2},
     outputSize = {256, 64},
     reduceSize = {128, 32, nil, nil},
     pool = nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1),
     batchNorm = true
   })

   -- Inception (4a)
   net:add(nn.Inception{
     inputSize = 640,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {192, 64},
     reduceSize = {96, 32, 128, 256},
     pool = nn.SpatialLPPooling(640, 2, 3, 3, 1, 1),
     batchNorm = true
   })

   -- Inception (4e)
   net:add(nn.Inception{
     inputSize = 640,
     kernelSize = {3, 5},
     kernelStride = {2, 2},
     outputSize = {256, 128},
     reduceSize = {160, 64, nil, nil},
     pool = nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1),
     batchNorm = true
   })

   -- Inception (5a)
   net:add(nn.Inception{
              inputSize = 1024,
              kernelSize = {3},
              kernelStride = {1},
              outputSize = {384},
              reduceSize = {96, 96, 256},
              pool = nn.SpatialLPPooling(960, 2, 3, 3, 1, 1),
              batchNorm = true
   })
   -- net:add(nn.Reshape(736,3,3))

   -- Inception (5b)
   net:add(nn.Inception{
              inputSize = 736,
              kernelSize = {3},
              kernelStride = {1},
              outputSize = {384},
              reduceSize = {96, 96, 256},
              pool = nn.SpatialMaxPooling(3, 3, 1, 1, 1, 1),
              batchNorm = true
   })

   net:add(nn.SpatialAveragePooling(3, 3))

   -- Validate shape with:
   -- net:add(nn.Reshape(736))

   net:add(nn.View(736))
   net:add(nn.Linear(736, 1024))
   net:add(nn.Normalize(2))

   return net
end

local function buildDeepConvnet()
    local convnet = nn.Sequential()
    -- input plane RGB 3x96x96
    convnet:add(cudnn.SpatialConvolution(3, 64, 3, 3))
    convnet:add(cudnn.SpatialBatchNormalization(64))
    convnet:add(nn.LeakyReLU(0.01))
    --convnet:add(cudnn.SpatialMaxPooling(2, 2, 1, 1))
    --convnet:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))
    convnet:add(cudnn.SpatialConvolution(64, 128, 3, 3))
    convnet:add(cudnn.SpatialBatchNormalization(128))
    convnet:add(nn.LeakyReLU(0.01))
    --convnet:add(cudnn.SpatialMaxPooling(2, 2, 1, 1))
    convnet:add(cudnn.SpatialConvolution(128, 256, 3, 3))
    convnet:add(cudnn.SpatialBatchNormalization(256))
    convnet:add(nn.LeakyReLU(0.01))
    convnet:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
    convnet:add(cudnn.SpatialConvolution(256, 512, 3, 3))
    convnet:add(cudnn.SpatialBatchNormalization(512))
    convnet:add(nn.LeakyReLU(0.01))
    --convnet:add(cudnn.SpatialMaxPooling(2, 2, 1, 1))
    convnet:add(cudnn.SpatialConvolution(512, 512, 3, 3))
    convnet:add(cudnn.SpatialBatchNormalization(512))
    convnet:add(nn.LeakyReLU(0.01))
    --convnet:add(cudnn.SpatialMaxPooling(2, 2, 1, 1))
    convnet:add(cudnn.SpatialConvolution(512, 512, 3, 3))
    convnet:add(cudnn.SpatialBatchNormalization(512))
    convnet:add(nn.LeakyReLU(0.01))
    convnet:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
    convnet:add(cudnn.SpatialConvolution(512, 1024, 3, 3))
    convnet:add(cudnn.SpatialBatchNormalization(1024))
    convnet:add(nn.LeakyReLU(0.01))
    --convnet:add(cudnn.SpatialMaxPooling(2, 2, 1, 1))
    convnet:add(cudnn.SpatialConvolution(1024, 1024, 3, 3))
    convnet:add(cudnn.SpatialBatchNormalization(1024))
    convnet:add(nn.LeakyReLU(0.01))
    --convnet:add(cudnn.SpatialMaxPooling(2, 2, 1, 1))
    convnet:add(cudnn.SpatialConvolution(1024, 2048, 3, 3))
    convnet:add(cudnn.SpatialBatchNormalization(2048))
    convnet:add(nn.LeakyReLU(0.01))
    --convnet:add(cudnn.SpatialMaxPooling(2, 2, 1, 1))
    convnet:add(cudnn.SpatialConvolution(2048, 2048, 3, 3))
    convnet:add(cudnn.SpatialBatchNormalization(2048))
    convnet:add(nn.LeakyReLU(0.01))
    convnet:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
    convnet:add(cudnn.SpatialConvolution(2048, 4096, 3, 3))
    convnet:add(cudnn.SpatialBatchNormalization(4096))
    convnet:add(nn.LeakyReLU(0.01))
    convnet:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
    weightInit(convnet, 'kaiming')
    
    local classifier = nn.Sequential()
    classifier:add(nn.Reshape(4096))
    classifier:add(nn.Linear(4096, 1000))
    classifier:add(nn.LeakyReLU(0.01))
    classifier:add(cudnn.BatchNormalization(1000))
    classifier:add(nn.Linear(1000, 256))
    classifier:add(nn.LeakyReLU(0.01))
    classifier:add(cudnn.BatchNormalization(256))
    classifier:add(nn.Linear(256, 2))
    classifier:add(cudnn.LogSoftMax())
    weightInit(classifier, 'kaiming')

    local model = nn.Sequential()
    model:add(convnet)
    model:add(classifier)
    local negWeight = 1.0 / (1.0 + opt.negExampleFactor)
    local loss = nn.ClassNLLCriterion(torch.Tensor{1.0 - negWeight, negWeight})
    print(model)
    return model, loss
end

local function buildResNet()
    local netOption = {depth = 101, shortcutType = 'B', dataset = 'imagenet', cudnn = 'fastest'}
    local resnet = createModel(netOption)
    resnet:add(cudnn.LogSoftMax())
    local negWeight = 1.0 / (1.0 + opt.negExampleFactor)
    return resnet, nn.ClassNLLCriterion(torch.Tensor{1.0 - negWeight, negWeight})
end

local model, loss = buildResNet()
model = makeDataParallel(model, opt.devCount)
if opt.type == 'cuda' then
    model = model:cuda()
    loss = loss:cuda()
end
return { model = model, loss = loss}

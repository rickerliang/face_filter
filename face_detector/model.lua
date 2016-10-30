require 'torch'
require 'nn'

local function classifier()
    local c = nn.Sequential()
    c:add(nn.Reshape(256))
    c:add(nn.Linear(256, 256))
    c:add(nn.LeakyReLU(0.01))
    c:add(nn.Linear(256, 2))
    c:add(cudnn.LogSoftMax())
    
    return c
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

local function simpleConvnetModel()
    print('building simple convnet model...')
    local convnet = nn.Sequential()
    -- input plane RGB 3x96x96
    convnet:add(cudnn.SpatialConvolution(3, 64, 4, 4))
    --convnet:add(cudnn.SpatialBatchNormalization(64))
    convnet:add(nn.LeakyReLU(0.01))
    convnet:add(cudnn.SpatialMaxPooling(4, 4, 3, 3))
    --convnet:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))

    convnet:add(cudnn.SpatialConvolution(64, 128, 4, 4))
    --convnet:add(cudnn.SpatialBatchNormalization(128))
    convnet:add(nn.LeakyReLU(0.01))
    convnet:add(cudnn.SpatialMaxPooling(4, 4, 3, 3))
    --convnet:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))

    convnet:add(cudnn.SpatialConvolution(128, 128, 4, 4))
    --convnet:add(cudnn.SpatialBatchNormalization(256))
    convnet:add(nn.LeakyReLU(0.01))
    convnet:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))

    convnet:add(cudnn.SpatialConvolution(128, 256, 2, 2))
    
    return convnet
end

local function buildModel()
    local convnet = simpleConvnetModel()
    local c = classifier()

    model = nn.Sequential()
    model:add(convnet)
    model:add(c)
    loss = nn.ClassNLLCriterion()
    print(model)
    return model, loss
end

local model, loss = buildModel()
if opt.type == 'cuda' then
    model = model:cuda()
    loss = loss:cuda()
end
return {model = model, loss = loss}

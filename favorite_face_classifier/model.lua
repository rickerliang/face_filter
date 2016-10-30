require 'torch'
require 'nn'

local function buildModel()
    print('building model...')
    local convnet = nn.Sequential()
    -- input plane RGB 3x96x96
    convnet:add(cudnn.SpatialConvolution(3, 16, 4, 4))
    convnet:add(cudnn.SpatialBatchNormalization(16))
    convnet:add(nn.LeakyReLU(0.01))
    convnet:add(cudnn.SpatialMaxPooling(4, 4, 3, 3))
    convnet:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))
    
    convnet:add(cudnn.SpatialConvolution(16, 32, 4, 4))
    convnet:add(cudnn.SpatialBatchNormalization(32))
    convnet:add(nn.LeakyReLU(0.01))
    convnet:add(cudnn.SpatialMaxPooling(4, 4, 3, 3))
    
    convnet:add(cudnn.SpatialConvolution(32, 32, 4, 4))
    convnet:add(cudnn.SpatialBatchNormalization(32))
    convnet:add(nn.LeakyReLU(0.01))
    convnet:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))

    convnet:add(cudnn.SpatialConvolution(32, 64, 2, 2))

    local classifier = nn.Sequential()
    classifier:add(nn.Reshape(64))
    classifier:add(nn.Linear(64, 256))
    classifier:add(nn.LeakyReLU(0.01))
    classifier:add(cudnn.BatchNormalization(256))
    classifier:add(nn.Linear(256, 2))
    classifier:add(cudnn.LogSoftMax())

    local model = nn.Sequential()
    model:add(convnet)
    model:add(classifier)
    loss = nn.ClassNLLCriterion()
    print(model)
    return model, loss
end

local model, loss = buildModel()
if opt.type == 'cuda' then
    model = model:cuda()
    loss = loss:cuda()
end
return { model = model, loss = loss}

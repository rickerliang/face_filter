require 'torch'
require 'image'
require 'nn'
require 'pl'
require 'trepl'

local function nilling(module)
    module.gradBias   = nil
    if module.finput then module.finput = torch.Tensor() end
    module.gradWeight = nil
    module.output     = torch.Tensor()
    module.fgradInput = nil
    module.gradInput  = nil
end

local function netLighter(network)
    nilling(network)
    if network.modules then
        for _,a in ipairs(network.modules) do
            netLighter(a)
        end
    end
end

local function saveTestFalseResult(trainEpoch, falsePos, falseNeg)
    paths.mkdir(trainEpoch .. '_test_falsePos')
    paths.mkdir(trainEpoch .. '_test_falseNeg')
    for _, img in ipairs(falsePos) do
        local falsePosImgName = paths.concat(
            trainEpoch .. '_test_falsePos', img[2] .. '.png')
        image.save(falsePosImgName, img[1])
    end
    for _, img in ipairs(falseNeg) do
        local falseNegImgName = paths.concat(
            trainEpoch .. '_test_falseNeg', img[2] .. '.png')
        image.save(falseNegImgName, img[1])
    end
end

local function saveTrainFalseResult(trainEpoch, falsePos, falseNeg)
    paths.mkdir(trainEpoch .. '_train_falsePos')
    paths.mkdir(trainEpoch .. '_train_falseNeg')
    for _, img in ipairs(falsePos) do
        local falsePosImgName = paths.concat(
            trainEpoch .. '_train_falsePos', img[2] .. '.png')
        image.save(falsePosImgName, img[1])
    end
    for _, img in ipairs(falseNeg) do
        local falseNegImgName = paths.concat(
            trainEpoch .. '_train_falseNeg', img[2] .. '.png')
        image.save(falseNegImgName, img[1])
    end
end
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> processing options')

opt = lapp[[
    -b, --batchSize          (default 128)         batch size
    -p, --type               (default cuda)        float or cuda(cudnn)
    -i, --devid              (default 1)           device ID (if using CUDA)
        --patches            (default 0.1)         percentage of samples to use for testing
        --visualize                                visualize dataset
        --smallTest                                use small data set to validate parameter tunning
        --stopStep           (default 10)          stop training while network performance never improve
        --normalizeSample    (default 0)           normalize each sample before training
        --modelSaveStep      (default 25)          save the best model after training step
]]

torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- type:
if opt.type == 'cuda' then
    print(sys.COLORS.red ..  '==> switching to CUDA')
    require 'cunn'
    require 'cudnn'
    cudnn.benchmark = true
    cutorch.setDevice(opt.devid)
    print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> load modules')

require 'data'
local data = getTrainingData()
print('data set size ' .. data:size())
local train = require 'train'
local test = require 'test'
if opt.visualize then
   -- Showing some training exaples
   image.display{image=data.data[{ {1,64} }], nrow=16, legend='Some training examples'}
   -- Showing some testing exaples
   image.display{image=data.testData[{ {1,64} }], nrow=16, legend='Some testing examples'}
end

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> training!')

local bestLoss = 100
local earlyStopStep = 1
local trainEpoch = 1
while true do
    if earlyStopStep > opt.stopStep then
        print(sys.COLORS.yellow .. '==> maximum step reached, stop training, best loss : ' .. bestLoss)
        os.rename('model/best_model.net' .. bestLoss, 'model/pretrained_model.net')
        
        local fileName = paths.concat('model', 'training_end_model.net')
        torch.save(fileName, model)
        return
    end
    collectgarbage()
    local _, trainFalsePos, trainFalseNeg = train(data)
    local currentLoss, falsePos, falseNeg = test(data)
    if currentLoss < bestLoss then
        print(sys.COLORS.yellow .. '==> previous loss ' .. bestLoss .. ' > ' ..  
            ' current loss ' .. currentLoss)
        earlyStopStep = 0
        if trainEpoch > opt.modelSaveStep then
            print(sys.COLORS.yellow .. '==> save best model')
            local bestModel = model:clone()
            paths.mkdir('model')
            os.remove('model/best_model.net' .. bestLoss)
            local fileName = paths.concat('model', 'best_model.net' .. currentLoss)
            --netLighter(bestModel)
            torch.save(fileName, bestModel)

            saveTestFalseResult(trainEpoch, falsePos, falseNeg)
            saveTrainFalseResult(trainEpoch, trainFalsePos, trainFalseNeg)
        end
        bestLoss = currentLoss
    end
    earlyStopStep = earlyStopStep + 1
    trainEpoch = trainEpoch + 1
end


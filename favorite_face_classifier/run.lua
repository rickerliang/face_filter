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

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> processing options')

opt = lapp[[
    -b, --batchSize          (default 128)         batch size
    -p, --type               (default cuda)        float or cuda
    -i, --devid              (default 1)           device ID (if using CUDA)
        --patches            (default 0.1)         percentage of samples to use for testing
        --visualize                                visualize dataset
        --smallTest                                use small data set to validate parameter tunning
        --stopStep           (default 30)          stop training while network performance never improve
]]

torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- type:
if opt.type == 'cuda' then
    print(sys.COLORS.red ..  '==> switching to CUDA')
    require 'cunn'
    require 'cudnn'
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
local m = require 'model'
local model = m.model
if opt.visualize then
   -- Showing some training exaples
   image.display{image=data.data[{ {1,64} }], nrow=16, legend='Some training examples'}
   -- Showing some testing exaples
   image.display{image=data.testData[{ {1,64} }], nrow=16, legend='Some testing examples'}
end

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> training!')

local bestFrr = 100
local earlyStopStep = 1
while true do
    if earlyStopStep > opt.stopStep then
        print(sys.COLORS.yellow .. '==> maximum step reached, stop training')
        os.rename('model/best_model.net', 'model/pretrained_model.net')
        return
    end
    collectgarbage()
    train(data)
    --local _, _, currentFrr, _, currentLoss = test(data)
    local currentLoss = test(data)
    --print(sys.COLORS.yellow .. '==> FRR : ' .. currentFrr[1] .. ' -- ' .. currentFrr[2])
    --if bestFrr > currentFrr[1] + currentFrr[2] then
    --    print(sys.COLORS.yellow .. '==> save best model')
    --    bestFrr = currentFrr[1] + currentFrr[2]
    --    earlyStopStep = 0
    --    local bestModel = model:clone()
    --    paths.mkdir('model')
    --    local fileName = paths.concat('model', 'best_model.net')
    --    torch.save(fileName, bestModel)
    --end
    if bestFrr > currentLoss then
        print(sys.COLORS.yellow .. '==> best loss ' .. bestFrr .. " > " .. currentLoss)
        print(sys.COLORS.yellow .. '==> save best model')
        bestFrr = currentLoss
        earlyStopStep = 0
        local bestModel = model:clone()
        paths.mkdir('model')
        local fileName = paths.concat('model', 'best_model.net')
        torch.save(fileName, bestModel)
    end
    earlyStopStep = earlyStopStep + 1
end


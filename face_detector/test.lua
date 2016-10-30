----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining some tools')

-- model:
local m = require 'model'
local model = m.model
local loss = m.loss

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix({'pos','neg'})
local testEpoch = 1

-- Batch test:

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining test procedure')

local function toFolder(inputs, targets, base)
    paths.mkdir('pos')
    paths.mkdir('neg')

    for i = 1, inputs:size(1) do
        if targets[i] == 1 then
            image.save('pos/' .. base + i .. '.png', inputs[i])
        else
            image.save('neg/' .. base + i .. '.png', inputs[i])
        end
    end
end

local function toFile(input, target, i)
    paths.mkdir('pos')
    paths.mkdir('neg')
    
    if target == 1 then
        print('tofile' .. 1)
        image.save('pos/' .. i .. '.png', input)
    else
        print('tofile' .. 2)
        image.save('neg/' .. i .. '.png', input)
    end
end

-- test function
local function test(data)
    -- local vars
    local time = sys.clock()
    paths.rmall('neg', 'yes')
    paths.rmall('pos', 'yes')
    
    local inputs = torch.Tensor(opt.batchSize, data.testData:size(2),
        data.testData:size(3), data.testData:size(4)) -- get size from data
    local targets = torch.Tensor(opt.batchSize)
    if opt.type == 'cuda' then
        inputs = inputs:cuda()
        targets = targets:cuda()
    end
    
    local lossTotal = 0
   
    -- test over test data
    print(sys.COLORS.red .. '==> testing on test set:')
    local batches = 0
    local falsePos = {}
    local falseNeg = {} 
    for t = 1, data:testSize(), opt.batchSize do
        -- batch fits?
        if (t + opt.batchSize - 1) > data:testSize() then
            xlua.progress(data:testSize(), data:testSize())
            -- -_-!!
            break
        end
        xlua.progress(t, data:testSize())
    
        -- create mini batch
        local idx = 1
        for i = t, t + opt.batchSize - 1 do
            inputs[idx] = data.testData[i]
            targets[idx] = data.testLabels[i]
            idx = idx + 1
        end
        
        --toFolder(inputs, targets, t - 1)
        
        -- test sample
        local preds = model:forward(inputs)
        lossTotal = lossTotal + loss:forward(preds, targets)
        
        -- confusion
        for i = 1, opt.batchSize do
            confusion:add(preds[i], targets[i])
            if torch.exp(preds[i][1]) > torch.exp(preds[i][2]) and targets[i] == 2 then
                local img = inputs[i]:clone()
                falsePos[#falsePos + 1] = {img, t + i - 1}
                --toFile(inputs[i], 1, t+i-1)
            elseif torch.exp(preds[i][1]) < torch.exp(preds[i][2]) and targets[i] == 1 then
                local img = inputs[i]:clone()
                falseNeg[#falseNeg + 1] = {img, t + i - 1}
                --toFile(inputs[i], 2, t+i-1)
            end
        end
        batches = batches + opt.batchSize
    end
    
    -- timing
    time = sys.clock() - time
    time = time / data:testSize()
    print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
    
    -- print confusion matrix
    print(confusion)
    local lossValue = lossTotal / batches
    print('==> loss ' .. lossValue)
    
    confusion:zero()
    testEpoch = testEpoch + 1
    return lossValue, falsePos, falseNeg
end

-- Export:
return test

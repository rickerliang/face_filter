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

-- Batch test:

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining test procedure')

-- test function
local function test(data)
    -- local vars
    model:evaluate()
    local time = sys.clock()
    confusion:zero()
    
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
    local batches = 1
    for t = 1, data:testSize(), opt.batchSize do
        -- disp progress
        xlua.progress(t, data:testSize())
        
        -- batch fits?
        if (t + opt.batchSize - 1) > data:testSize() then
            break
        end
        
        -- create mini batch
        local idx = 1
        for i = t,t+opt.batchSize-1 do
            inputs[idx] = data.testData[i]
            targets[idx] = data.testLabels[i]
            idx = idx + 1
        end
        
        -- test sample
        local preds = model:forward(inputs)
        lossTotal = lossTotal + loss:forward(preds, targets)
        
        -- confusion
        confusion:batchAdd(preds, targets)
        batches = t
    end
    
    -- timing
    time = sys.clock() - time
    time = time / data:testSize()
    print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
    
    -- print confusion matrix
    print(confusion)
    local lossValue = lossTotal / batches
    print('==> loss ' .. lossValue)
    
    --return confusion:farFrr(), lossValue
    return lossValue
end

-- Export:
return test

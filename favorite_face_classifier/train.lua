require 'torch'
require 'xlua'
require 'optim'

local m = require 'model'
local model = m.model
local loss = m.loss
local w, dE_dw = model:getParameters()
--w:uniform(-0.08, 0.08)
local batchSize = opt.batchSize
local confusion = optim.ConfusionMatrix({'pos','neg'})
local epoch

-- trainData{data, label, size}
local function train(trainData)
    
    -- epoch tracker
    model:training()
    epoch = epoch or 1
    confusion:zero()
    
    -- local vars
    local time = sys.clock()
    
    -- shuffle at each epoch
    local shuffle = torch.randperm(trainData:size())
    
    -- do one epoch
    print(sys.COLORS.green .. '==> doing epoch on training data:')
    print("==> epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
    
    local x = torch.Tensor(batchSize,trainData.data:size(2),
        trainData.data:size(3), trainData.data:size(4)) --faces data
    local yt = torch.Tensor(batchSize)
    if opt.type == 'cuda' then
        x = x:cuda()
        yt = yt:cuda()
    end

    local lossTotal = 0
    
    local batch = 1
    for t = 1, trainData:size(), batchSize do
        -- disp progress
        xlua.progress(t, trainData:size())
        collectgarbage()
        
        -- batch fits?
        if (t + batchSize - 1) > trainData:size() then
            break
        end
        
        -- create mini batch
        local idx = 1
        for i = t, t + batchSize - 1 do
            x[idx] = trainData.data[shuffle[i]]
            yt[idx] = trainData.labels[shuffle[i]]
            idx = idx + 1
        end

        -- create closure to evaluate f(X) and df/dX
        local eval_E = function(w)
            -- reset gradients
            dE_dw:zero()
            
            -- evaluate function for complete mini batch
            local y = model:forward(x)
            local E = loss:forward(y, yt)
            lossTotal = lossTotal + E
            
            -- estimate df/dW
            local dE_dy = loss:backward(y, yt)
            model:backward(x, dE_dy)
            
            -- update confusion
            confusion:batchAdd(y, yt)
            
            -- return f and df/dX
            return E, dE_dw
        end
        
        -- optimize on current mini-batch
        optim.adam(eval_E, w)
        batches = t
    end
    
    -- time taken
    time = sys.clock() - time
    time = time / trainData:size()
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
    
    -- print confusion matrix
    print(confusion)
    local lossValue = lossTotal / batches
    print('==> loss ' .. lossValue)
    
    -- next epoch
    epoch = epoch + 1
    
    return lossValue
end

return train


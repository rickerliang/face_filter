require 'torch'
require 'xlua'
require 'optim'

local m = require 'model'
local model = m.model
local loss = m.loss
local w, dE_dw = model:getParameters()
local batchSize = opt.batchSize
local confusion = optim.ConfusionMatrix({'pos','neg'})
local epoch

-- trainData{data, label, size}
local function train(trainData)
    
    -- epoch tracker
    epoch = epoch or 1
    
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
    
    local batches = 0
    local falsePos = {}
    local falseNeg = {} 
    for t = 1, trainData:size(), batchSize do
        -- disp progress
        
        collectgarbage()
        
        -- batch fits?
        if (t + batchSize - 1) > trainData:size() then
            xlua.progress(trainData:size(), trainData:size())
            -- -_-!!
            break;
        end
        xlua.progress(t, trainData:size())
        -- create mini batch
        local idx = 1
        for i = t, t + batchSize - 1 do
            x[idx] = trainData.data[shuffle[i]]
            yt[idx] = trainData.labels[shuffle[i]]
            idx = idx + 1
        end

        batches = batches + batchSize
        
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
            for i = 1, batchSize do
                confusion:add(y[i], yt[i])
                if torch.exp(y[i][1]) > torch.exp(y[i][2]) and yt[i] == 2 then
                    local img = x[i]:clone()
                    falsePos[#falsePos + 1] = {img, t + i - 1}
                elseif torch.exp(y[i][1]) < torch.exp(y[i][2]) and yt[i] == 1 then
                    local img = x[i]:clone()
                    falseNeg[#falseNeg + 1] = {img, t + i - 1}
                end
            end
            
            -- return f and df/dX
            return E, dE_dw
        end
        
        -- optimize on current mini-batch
        optim.rmsprop(eval_E, w)
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
    confusion:zero()
    epoch = epoch + 1
    
    return lossValue, falsePos, falseNeg
end

return train


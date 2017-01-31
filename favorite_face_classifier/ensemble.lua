require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'paths'
require 'nngraph'
require 'inception'
require 'resnet'

local function endsWith(String, End)
   return End =='' or string.sub(String, -string.len(End)) == End
end

-- building vote graph
-- pred table of each model probability output 
-- like { n x 2, n x 2, n x 2 }, where n x 2 like (after exp) [0.3 0.7, 0.4 0.5]
-- then translate to vote according *threshold*
-- like [0.6, 0.4] change to [1, 0], where threshold = 0.6
-- another example [0.4, 0.6] change to [1, 0], where threshold = 0.4
-- then sum and average by *modelCount*
-- example:
-- pred : {[0.3 0.7, 0.4 0.6], [0.2 0.8, 0.6 0.4], [0.6 0.4, 0.3 0.7]}
-- threshold : 0.6, modelCount 3
-- translate to {[0 1, 0 1], [0 1, 1 0], [1 0, 0 1]}
-- sum {[1 2, 1, 2]}
-- average {[0.33 0.66, 0.33 0.66]}
local function probabilityToVote(input, pred, threshold, modelCount)
    local thresholdModule = nn.Sequential()
    thresholdModule:add(nn.Select(2, 1))
    thresholdModule:add(nn.Exp())
    thresholdModule:add(nn.Threshold(threshold, 2))
    thresholdModule:add(nn.Threshold(1.1, 1))
    thresholdModule:add(nn.MulConstant(-1))
    thresholdModule:add(nn.AddConstant(2))
    local identity = nn.Sequential():add(nn.View(-1, 1))
    local oneMinusThreshold = nn.Sequential()
    oneMinusThreshold:add(nn.MulConstant(-1))
    oneMinusThreshold:add(nn.AddConstant(1))
    oneMinusThreshold:add(nn.View(-1, 1))
    local concat = nn.ConcatTable()
    concat:add(identity)
    concat:add(oneMinusThreshold)
    local join = nn.JoinTable(2)
    local wrap = nn.Sequential()
    wrap:add(thresholdModule)
    wrap:add(concat)
    wrap:add(join)

    local tableOutput = nn.MapTable(wrap)(pred)
    local sum = nn.CAddTable()(tableOutput)
    local output = nn.MulConstant((1.0 / modelCount), true)(sum)
    local vote = nn.gModule({input}, {output})
    return vote
end

local function buildEnsembleGraph(modelsDir, threshold)
    print('build ensemble graph, model dir : ' .. modelsDir ..
        ', threshold : ' .. threshold)
    local modelIter = paths.iterfiles(modelsDir)
    --local modelOutput = {}
    local modelCount = 0
    
    local input = nn.Identity()()
    local concat = nn.ConcatTable()
    while true do
        local file = modelIter()
        if file == nil then
            break
        end

        local modelFile = paths.concat(modelsDir, file)
        if endsWith(modelFile, '.net') then
            --print(modelFile)
            local model = torch.load(modelFile)
            --print( torch.typename(model))
            if  torch.typename(model) == 'nn.Sequential' then
                --table.insert(modelOutput, model(input))
                concat:add(model)
                modelCount = modelCount + 1
            elseif  torch.typename(model) == 'nn.DataParallelTable' then
                --table.insert(modelOutput, model:get(1)(input))
                concat:add(model:get(1))
                modelCount = modelCount + 1
            end
        end
    end
    print('model count ' .. modelCount)
    assert(modelCount > 0)
    local pred = concat(input)
    local vote = probabilityToVote(input, pred, threshold, modelCount)
    return vote
end



return buildEnsembleGraph
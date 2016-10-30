require 'image'
require 'torch'
require 'nms'
require 'nn'

-- A slower implementation

cmd = torch.CmdLine()
cmd:text()
cmd:text('Detect a face using slide window')
cmd:text()
cmd:text('Options')
-- optional parameters
cmd:option('-input', 'pic_to_detect', 'input dir pic to detect a face')
cmd:option('-step', 15, 'sliding window step in pixel')
cmd:option('-model', 'model/pretrained_model.net', 'face detect model')
cmd:option('-output', 'face_output', 'face cut output dir')
cmd:option('-type', 'cuda', 'model type cuda(cudnn) or cpu')
cmd:option('-devid', 1, 'cuda device id')
cmd:option('-batchSize', 256, 'prediction batch size')
cmd:option('-threshold', 0.99, 'prediction threshold')
cmd:option('-excludeSize', 120, 'exclude image while size less then')
cmd:option('-verbose', 0, 'output intermediate face cut')
cmd:option('-prune', 'nms', 'face bounding box prune method : nms or simple')
cmd:option('-normalizeInput', 0, 'normalize input picture before classify. note if model train with normalize input, set this option to 1, otherwise 0')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
opt.output = opt.output .. opt.threshold

torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')
if opt.type == 'cuda' then
    print(sys.COLORS.cyan ..  'switching to CUDA')
    require 'cunn'
    require 'cudnn'
    cudnn.benchmark = true
    cutorch.setDevice(opt.devid)
    print(sys.COLORS.cyan ..  'using GPU #' .. cutorch.getDevice())
end
local model = torch.load(opt.model)
if opt.type == 'cuda' then model = model:cuda() end
print(sys.COLORS.green ..  'processing file in ' .. opt.input)

local function tableToTensor(table)
  local tensorSize = table[1]:size()
  local tensorSizeTable = {-1}
  for i=1,tensorSize:size(1) do
    tensorSizeTable[i+1] = tensorSize[i]
  end
  local merge = nn.Sequential()
    :add(nn.JoinTable(1))
    :add(nn.View(unpack(tensorSizeTable)))

  return merge:forward(table)
end

local function tableSlice(tbl, first, last)
  local sliced = {}

  for i = first, last do
    sliced[#sliced+1] = tbl[i]
  end

  return sliced
end

local function prune(detections)
    local pruned = {}

    local index = 1
    for i,detect in ipairs(detections) do
        local duplicate = 0
        for j, prune in ipairs(pruned) do
            if torch.abs(prune.x-detect.x)<=60 or torch.abs(prune.y-detect.y)<=60 then
                duplicate = 1
            end
        end

        if duplicate == 0 then
            pruned[index] = {x=detect.x, y=detect.y, w=detect.w, h=detect.h}
            index = index+1
        end 
    end

    return pruned
end

local channels = {'r', 'g', 'b'}
local function normalizeRGB(image)
    for i,channel in ipairs(channels) do
        -- normalize each channel globally:
        local mean = image[{ i,{},{} }]:mean()
        local std = image[{ i,{},{} }]:std()
        image[{ i,{},{} }]:add(-mean)
        image[{ i,{},{} }]:div(std + 1e-8)
        --print('image data, '..channel..'-channel, mean:               ' .. image[{i, {}, {}}]:mean())
        --print('image data, '..channel..'-channel, standard deviation: ' .. image[{i, {}, {}}]:std())
    end
    return image
end

-- return
--  slices : scaled and normalized image slices
--  rgbSlices : non scaled rgb image slices
--  slicesCoords : non scaled image slice coord - x, y, w, h
--  slicesVertices : non scaled image slice vertices - x1, y1, x2, y2
local function prepare(inputImage, scaleFactor)
    local y = inputImage:clone()
    print(sys.COLORS.Green ..  '====> scale factor ' .. scaleFactor) 
    --normalize
    print(sys.COLORS.Green ..  '====> normalizing input...')
    
    if opt.normalizeInput == 1 then
        local channels = {'r', 'g', 'b'}
        
        for i,channel in ipairs(channels) do
            -- normalize each channel globally:
            local mean = y[{ i,{},{} }]:mean()
            local std = y[{ i,{},{} }]:std() + 1e-8
            y[{ i,{},{} }]:add(-mean)
            y[{ i,{},{} }]:div(std)
            
            --print('mean ' .. channel .. ' ' .. mean[i] .. ' -> ' .. y[{ i,{},{} }]:mean())
            --print('std ' .. channel .. ' ' .. std[i] .. ' -> ' .. y[{ i,{},{} }]:std())
        end
    end
    
    print('input dimension CxWxH : ' .. y:size()[1] .. 'x' .. y:size()[3] .. 'x' .. y:size()[2])
    local slices = {}
    local rgbSlices = {}
    local slicesCoords = {}
    local slicesVertics = {}

    local i = 1
    local j = 1
    local scaledImage = y
    if scaleFactor ~= 1 then
        scaledImage = image.scale(y, y:size()[3] * scaleFactor,
            y:size()[2] * scaleFactor, simple)
    end
    while i + 96 <= scaledImage:size()[2]  do
        while j + 96 <= scaledImage:size()[3] do
            local slice = scaledImage[{{}, {i, i + 96}, {j, j + 96}}]
            table.insert(slices, slice)
            local rgbSlice = inputImage[{{}, {i / scaleFactor, (i + 96) / scaleFactor},
                {j / scaleFactor, (j + 96) / scaleFactor}}]
            table.insert(rgbSlices, rgbSlice)
            table.insert(
                slicesCoords, {x = j / scaleFactor, y = i / scaleFactor,
                    w = 96 / scaleFactor, h = 96 / scaleFactor})
            local vertices = torch.Tensor(1, 4)
            vertices[1][1] = j / scaleFactor
            vertices[1][2] = i / scaleFactor
            vertices[1][3] = (j + 96) / scaleFactor
            vertices[1][4] = (i + 96) / scaleFactor
            table.insert(slicesVertics, vertices)
            j = j + opt.step
        end
        j = 1
        i = i + opt.step
    end
    print('scale dimension CxWxH : ' .. scaledImage:size()[1] .. 'x' ..
        scaledImage:size()[3] .. 'x' .. scaledImage:size()[2])
        
    return slices, rgbSlices, slicesCoords, slicesVertics
end

-- return
--  detectAFace : true or false
--  faceIndices : table contains indices in slices
--  faceProbability : table contains faces probability, table size=#faceIndices
local function detect(slices, rgbSlices, inputImage, scaleFactor, file)
    if #slices <= 0 then
        return false
    end        
    paths.mkdir(opt.output)
    
    local detectAFace = false
    local faceIndices = {}
    local faceProbability = {}
    local probability = opt.threshold
    print('slices : '.. #slices .. ' batchSize : ' .. opt.batchSize ..
        ' probability : ' .. probability)
    local i = 1
    while i + opt.batchSize - 1 < #slices do
        print(i .. ' - ' .. i + opt.batchSize - 1)
        subTable = tableSlice(slices, i, i + opt.batchSize - 1)
        local s = torch.Tensor(tableToTensor(subTable))
        s = s:cuda()
        local result = model:forward(s)
        result = torch.exp(result)
        for j = 1, result:size()[1] do
            if result[j][1] >= probability then
                if opt.verbose ~= 0 then
                    image.save(paths.concat(opt.output,
                        paths.basename(file) .. '_fc_' ..
                        string.format('%.4f', result[j][1])) .. '_' .. i .. '_' .. j ..
                        '_' .. scaleFactor .. '.png', rgbSlices[i + j - 1])
                end
                table.insert(faceIndices, i + j - 1)
                local p = torch.Tensor(1)
                p:fill(result[j][1])
                table.insert(faceProbability, p)
                detectAFace = true
            end
        end
        i = i + opt.batchSize
    end
    
    local mod = #slices % opt.batchSize
    print(i .. ' - ' .. i + mod)
    local subTable = tableSlice(slices, i, i + mod)
    local s = torch.Tensor(tableToTensor(subTable))
    s = s:cuda()
    local result = model:forward(s)
    result = torch.exp(result)
    for j = 1, result:size()[1] do
        if result[j][1] >= probability then
            if opt.verbose ~= 0 then
                image.save(paths.concat(opt.output,
                    paths.basename(file) .. '_fc_' .. string.format('%.4f', result[j][1])) ..
                    '_' .. i .. '_' .. j .. '_' .. scaleFactor .. '.png', rgbSlices[i + j - 1])
            end
            table.insert(faceIndices, i + j - 1)
            local p = torch.Tensor(1)
            p:fill(result[j][1])
            table.insert(faceProbability, p)
            detectAFace = true
        end
    end
    
    print('detect result ' .. tostring(detectAFace))
    return detectAFace, faceIndices, faceProbability
end

local function run(file)
    local status, inputImage = pcall(image.load, file, 3)
    if status == false then
        print(sys.COLORS.red ..  '====> load image error ' .. file)
        return false
    end
    
    if inputImage:size(3) <= opt.excludeSize or inputImage:size(2) <= opt.excludeSize then
        local failPath = paths.concat(opt.output, 'exclude_by_size_' .. opt.excludeSize)
        paths.mkdir(failPath)
        local savePath = paths.concat(failPath, paths.basename(file))
        image.save(savePath, inputImage)
        return
    end
    
    local scaleFactors = {}    
    if inputImage:size(3) <= 300 or inputImage:size(2) <= 300 then
        table.insert(scaleFactors, 1.5)
        table.insert(scaleFactors, 1)
        table.insert(scaleFactors, 0.75)
        table.insert(scaleFactors, 0.5)
    elseif inputImage:size(3) <= 800 or inputImage:size(2) <= 800 then
        table.insert(scaleFactors, 1.2)
        table.insert(scaleFactors, 1)
        table.insert(scaleFactors, 0.75)
        table.insert(scaleFactors, 0.5)
        table.insert(scaleFactors, 0.25)
        table.insert(scaleFactors, 0.17)
    elseif inputImage:size(3) < 1300 or inputImage:size(2) < 1300 then
        table.insert(scaleFactors, 1)
        table.insert(scaleFactors, 0.75)
        table.insert(scaleFactors, 0.5)
        table.insert(scaleFactors, 0.25)
        table.insert(scaleFactors, 0.125)
    elseif inputImage:size(3) < 2400 or inputImage:size(2) < 2400 then
        table.insert(scaleFactors, 0.75)
        table.insert(scaleFactors, 0.5)
        table.insert(scaleFactors, 0.25)
        table.insert(scaleFactors, 0.125)
        table.insert(scaleFactors, 0.06)
    else
        table.insert(scaleFactors, 0.5)
        table.insert(scaleFactors, 0.25)
        table.insert(scaleFactors, 0.125)
        table.insert(scaleFactors, 0.06)
        table.insert(scaleFactors, 0.03)
    end
    local hasFace = 0
    local facesCoords = {}
    local facesVertices = {}
    local facesProbability = {}
    for i,s in ipairs(scaleFactors) do
        local slices, rgbSlices, slicesCoords, slicesVertices = prepare(inputImage, s)
        local faceDetected, faceIndices, p =
            detect(slices, rgbSlices, inputImage, s, file)
        if faceDetected then
            hasFace = hasFace + 1
            for j, index in ipairs(faceIndices) do
                table.insert(facesCoords, slicesCoords[index])
                table.insert(facesVertices, slicesVertices[index])
                table.insert(facesProbability, p[j])
            end
        end
    end

    print (sys.COLORS.Yellow ..  '====> has face flag: ' .. hasFace)
    if hasFace == 0 then
        local failPath = paths.concat(opt.output, 'fail_to_detect_at_first_time')
        paths.mkdir(failPath)
        local savePath = paths.concat(failPath, paths.basename(file))
        image.save(savePath, inputImage)
        return
    end
    
    if opt.prune == 'simple' then
        -- a simple method to prune faces bounding box
        local prunedCoords = prune(facesCoords)
        local prunePath = paths.concat(opt.output, 'pruned')
        paths.mkdir(prunePath)
        for _, coord in ipairs(prunedCoords) do
            local s = inputImage[{{}, {coord.y, coord.y + coord.h}, {coord.x, coord.x + coord.w}}]
            local savePath = paths.concat(prunePath, paths.basename(file) .. coord.x .. '_' .. coord.y .. '.png')
            image.save(savePath, s)
        end
    else    
        -- use Non-Maximum Suppression to prune faces bounding box
        local verticesTensor = tableToTensor(facesVertices)
        local probabilityTensor = tableToTensor(facesProbability)
        verticesTensor = torch.squeeze(verticesTensor, 2)
        -- a tensor Nx4
        verticesTensor = torch.cat(verticesTensor, probabilityTensor, 2)
        -- a tensor Nx5, last dim contains faces probability
        -- nms according to probability (last dim)
        local indicesRemain = nms(verticesTensor, 0.1, 5)
        local nmsPath = paths.concat(opt.output, 'nms')
        paths.mkdir(nmsPath)
        for i = 1, indicesRemain:size()[1] do
            local verticesIndex = indicesRemain[i]
            local s = inputImage[{{}, {verticesTensor[verticesIndex][2], verticesTensor[verticesIndex][4]},
                {verticesTensor[verticesIndex][1], verticesTensor[verticesIndex][3]}}]
            local savePath = paths.concat(nmsPath, paths.basename(file) ..
                verticesTensor[verticesIndex][1] .. '_' .. verticesTensor[verticesIndex][2] .. '_' ..
                verticesTensor[verticesIndex][5] .. '.png')
            image.save(savePath, s)
        end
    end
    
    return true
end

print(sys.COLORS.Yellow ..  '====> begin')
for file in paths.iterfiles(opt.input) do
    local f = paths.concat(opt.input, file)
    print(sys.COLORS.Green ..  '====> ' .. file)
    --local status, result = pcall(run ,f)
    run(f)
    if false then
        local errorPath = paths.concat(opt.output, 'error_to_detect')
        paths.mkdir(errorPath)
        local status, i = pcall(image.load, f)
        if status then
            local savePath = paths.concat(errorPath, paths.basename(file))
            image.save(savePath, i)
        else
            print(sys.COLORS.red ..  '====> load image error ' .. f)
        end
    end
end
print(sys.COLORS._yellow ..  '====> end')

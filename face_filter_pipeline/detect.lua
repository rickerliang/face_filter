require 'image'
require 'torch'
require '../face_detector/nms'
require 'nn'

torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')
--if opt.type == 'cuda' then
    print(sys.COLORS.cyan ..  'switching to CUDA')
    require 'cunn'
    require 'cudnn'
    cudnn.benchmark = true
    cutorch.setDevice(opt.devid)
    print(sys.COLORS.cyan ..  'using GPU #' .. cutorch.getDevice())
--end
print(sys.COLORS.green ..  'loading detector model ')
local modelDetector = torch.load('../face_detector/model/pretrained_model.net')
--if opt.type == 'cuda' then 
    modelDetector = modelDetector:cuda()
--end
print(sys.COLORS.green ..  'loading classifier model ')
local modelClassifier
if opt.vgg == true then
    modelClassifier = torch.load('../favorite_face_classifier/model/pretrained_model_vgg.net')
else
    modelClassifier = torch.load('../favorite_face_classifier/model/pretrained_model.net')
end
local excludeSize = 120
local batchSize = 64
local slideStep = 15
local frameWidth = 96
local frameHeight = 96

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

-- return
--  detectAFace : true or false
--  faceIndices : table contains indices in slices
--  faceProbability : table contains faces probability, table size=#faceIndices
local function detect(slices, threshold)
    if #slices <= 0 then
        return false
    end        

    local detectAFace = false
    local faceIndices = {}
    local faceProbability = {}
    local probability = threshold
    --print('slices : '.. #slices .. ' batchSize : ' .. batchSize ..
        --' probability : ' .. probability)
    local i = 1
    while i + batchSize - 1 < #slices do
        --print(i .. ' - ' .. i + batchSize - 1)
        subTable = tableSlice(slices, i, i + batchSize - 1)
        local s = torch.Tensor(tableToTensor(subTable))
        s = s:cuda()
        local result = modelDetector:forward(s)
        result = torch.exp(result)
        for j = 1, result:size()[1] do
            if result[j][1] >= probability then
                table.insert(faceIndices, i + j - 1)
                local p = torch.Tensor(1)
                p:fill(result[j][1])
                table.insert(faceProbability, p)
                detectAFace = true
            end
        end
        i = i + batchSize
    end
    
    local mod = #slices % batchSize
    --print(i .. ' - ' .. i + mod)
    local subTable = tableSlice(slices, i, i + mod)
    local s = torch.Tensor(tableToTensor(subTable))
    s = s:cuda()
    local result = modelDetector:forward(s)
    result = torch.exp(result)
    for j = 1, result:size()[1] do
        if result[j][1] >= probability then
            table.insert(faceIndices, i + j - 1)
            local p = torch.Tensor(1)
            p:fill(result[j][1])
            table.insert(faceProbability, p)
            detectAFace = true
        end
    end
    
    --print('detect result ' .. tostring(detectAFace))
    return detectAFace, faceIndices, faceProbability
end

local function prepare(inputImage, scaleFactor)
    local y = inputImage:clone()
    --print(sys.COLORS.Green ..  '====> scale factor ' .. scaleFactor) 
    
    --print('input dimension CxWxH : ' .. y:size()[1] .. 'x' .. y:size()[3] .. 'x' .. y:size()[2])
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
            j = j + slideStep
        end
        j = 1
        i = i + slideStep
    end
    --print('scale dimension CxWxH : ' .. scaledImage:size()[1] .. 'x' ..
        --scaledImage:size()[3] .. 'x' .. scaledImage:size()[2])
        
    return slices, rgbSlices, slicesCoords, slicesVertics
end

local function detectFace(imageTensor, detectorThreshold)
    if imageTensor:size(3) <= excludeSize or imageTensor:size(2) <= excludeSize then
        print(sys.COLORS.cyan ..  'image less than ' .. excludeSize)
        return false, nil
    end
    
    local scaleFactors = {}    
    if imageTensor:size(3) <= 300 or imageTensor:size(2) <= 300 then
        table.insert(scaleFactors, 4)
        table.insert(scaleFactors, 3)
        table.insert(scaleFactors, 2)
        table.insert(scaleFactors, 1)
        table.insert(scaleFactors, 0.75)
        table.insert(scaleFactors, 0.5)
    elseif imageTensor:size(3) <= 800 or imageTensor:size(2) <= 800 then
        table.insert(scaleFactors, 1.2)
        table.insert(scaleFactors, 1)
        table.insert(scaleFactors, 0.75)
        table.insert(scaleFactors, 0.5)
        table.insert(scaleFactors, 0.25)
    elseif imageTensor:size(3) < 1300 or imageTensor:size(2) < 1300 then
        table.insert(scaleFactors, 1)
        table.insert(scaleFactors, 0.75)
        table.insert(scaleFactors, 0.5)
        table.insert(scaleFactors, 0.25)
    elseif imageTensor:size(3) < 2400 or imageTensor:size(2) < 2400 then
        table.insert(scaleFactors, 0.75)
        table.insert(scaleFactors, 0.5)
        table.insert(scaleFactors, 0.25)
        table.insert(scaleFactors, 0.125)
    else
        table.insert(scaleFactors, 0.5)
        table.insert(scaleFactors, 0.25)
        table.insert(scaleFactors, 0.125)
        table.insert(scaleFactors, 0.06)
    end
    
    local hasFace = 0
    local facesCoords = {}
    local facesVertices = {}
    local facesProbability = {}
    for i,s in ipairs(scaleFactors) do
        local slices, rgbSlices, slicesCoords, slicesVertices = prepare(imageTensor, s)
        local faceDetected, faceIndices, p =
            detect(slices, detectorThreshold)
        if faceDetected then
            hasFace = hasFace + 1
            for j, index in ipairs(faceIndices) do
                table.insert(facesCoords, slicesCoords[index])
                table.insert(facesVertices, slicesVertices[index])
                table.insert(facesProbability, p[j])
            end
        end
    end
    
    if hasFace == 0 then
        --print(sys.COLORS.cyan ..  'image no face ')
        return false, nil
    end
    
    local verticesTensor = tableToTensor(facesVertices)
    local probabilityTensor = tableToTensor(facesProbability)
    verticesTensor = torch.squeeze(verticesTensor, 2)
    -- a tensor Nx4
    verticesTensor = torch.cat(verticesTensor, probabilityTensor, 2)
    -- a tensor Nx5, last dim contains faces probability
    -- nms according to probability (last dim)
    local indicesRemain = nms(verticesTensor, 0.1, 5)
    
    local faces = {}
    for i = 1, indicesRemain:size()[1] do
        local verticesIndex = indicesRemain[i]
        local s = imageTensor[{{}, {verticesTensor[verticesIndex][2], verticesTensor[verticesIndex][4]},
            {verticesTensor[verticesIndex][1], verticesTensor[verticesIndex][3]}}]
        table.insert(faces, {s:clone(), {verticesTensor[verticesIndex][2], verticesTensor[verticesIndex][4],
            verticesTensor[verticesIndex][1], verticesTensor[verticesIndex][3]}})
    end
    
    if #faces > 0 then
        return true, faces
    else
        --print(sys.COLORS.cyan ..  'image no face ')
        return false, nil
    end
end

local function classify(images, classifierThreshold)
    if #images <= 0 then
        return false
    end
    --print('slices ' .. #images)
    local likeImagesIndices = {}
    local probability = classifierThreshold
    local i = 1
    while i + batchSize - 1 < #images do
        --print(i .. ' - ' .. i + batchSize - 1)
        subTable = tableSlice(images, i, i + batchSize - 1)
        local s = torch.Tensor(tableToTensor(subTable))
        s = s:cuda()
        local result = modelClassifier:forward(s)
        result = torch.exp(result)
        for j = 1, result:size()[1] do
            if result[j][1] >= probability then
                table.insert(likeImagesIndices, i + j - 1)
            end
        end
        i = i + batchSize
    end
    
    local mod = #images % batchSize
    --print(i .. ' - ' .. i + mod)
    local subTable = tableSlice(images, i, i + mod)
    local s = torch.Tensor(tableToTensor(subTable))
    s = s:cuda()
    local result = modelClassifier:forward(s)
    result = torch.exp(result)
    for j = 1, result:size()[1] do
        if result[j][1] >= probability then
            table.insert(likeImagesIndices, i + j - 1)
        end
    end
    
    return likeImagesIndices
end

function pipeline(imageTensor, detectorThreshold, classifierThreshold)
    local hasFace, faces = detectFace(imageTensor, detectorThreshold)
    
    if not hasFace then
        print(sys.COLORS.red ..  'no face')
        return false, nil
    end
    
    local scaledImages = {}
    for i, imageInfo in ipairs(faces) do
        local imageTensor = imageInfo[1]
        if imageTensor:size()[2] ~= frameHeight and imageTensor:size()[3] ~= frameWidth then
            imageTensor = image.scale(imageTensor, frameWidth, frameHeight)
            table.insert(scaledImages, imageTensor)
        else
            table.insert(scaledImages, imageInfo[1])
        end
    end
    
    local ret = imageTensor:clone()
    
    local beautyFaceIndices = classify(scaledImages, classifierThreshold)
    if #beautyFaceIndices > 0 then
        local retFaces = {}
        for j, indices in ipairs(beautyFaceIndices) do
            local vetices = faces[indices][2]
            print('bb x1:' .. vetices[3] .. ' y1:' .. vetices[1] .. ' x2:'
                .. vetices[4] .. ' y2:' .. vetices[2])
            local retFace = ret[{{}, {vetices[1] + 1, vetices[2] - 1}, {vetices[3] + 1, vetices[4] - 1}}]:clone()
            retFaces[#retFaces + 1] = retFace
            ret = image.drawRect(ret, vetices[3] + 5, vetices[1] + 5, vetices[4] - 5, vetices[2] - 5,
                {lineWidth = 4, color = {0, 255, 0}})
        end
        return true, ret, retFaces
    else
        for i, _ in ipairs(faces) do
            local vetices = faces[i][2]
                ret = image.drawRect(ret, vetices[3] + 5, vetices[1] + 5, vetices[4] - 5, vetices[2] - 5,
                    {lineWidth = 5, color = {0, 0, 255}})
        end
        print(sys.COLORS.red ..  'no beauty face')
        return false, ret
    end
end

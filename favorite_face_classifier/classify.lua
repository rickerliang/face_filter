require 'image'
require 'torch'
require 'inception'

local channels = {'r', 'g', 'b'}
local frameWidth = 96
local frameHeight = 96

cmd = torch.CmdLine()
cmd:text()
cmd:text('Classify a face')
cmd:text()
cmd:text('Options')
-- optional parameters
cmd:option('-input', 'pic_to_classify', 'pictures folder to classify a face you like or not')
cmd:option('-model', 'model/pretrained_model.net', 'classifier model')
cmd:option('-output', 'classify_output', 'face classify output dir')
cmd:option('-type', 'cuda', 'model type cuda or cpu')
cmd:option('-devid', 1, 'cuda device id')
cmd:option('-batchSize', 64, 'prediction batch size')
cmd:option('-threshold', 0.99, 'prediction threshold')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')
if opt.type == 'cuda' then
    print(sys.COLORS.cyan ..  'switching to CUDA')
    require 'cunn'
    require 'cudnn'
    cutorch.setDevice(opt.devid)
    print(sys.COLORS.cyan ..  'using GPU #' .. cutorch.getDevice())
end
print(sys.COLORS.green ..  'processing file in ' .. opt.input)
local images = {}
local fileNames = {}
local originalImages={}

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

local function normalize(inputImage)
    local y = inputImage:clone()
    local channels = {'r', 'g', 'b'}
    
    for i,channel in ipairs(channels) do
        local mean = {}
        local std = {}
        mean[i] = y[{ i,{},{} }]:mean()
        std[i] = y[{ i,{},{} }]:std()
        y[{ i,{},{} }]:add(-mean[i])
        y[{ i,{},{} }]:div(std[i])
    end
    return y
end

local function classify(slices, origins, files)
    if #slices <= 0 then
        return false
    end
    print('slices ' .. #slices)
    paths.mkdir(opt.output .. opt.threshold)
    local likePath = paths.concat(opt.output .. opt.threshold, 'like')
    paths.mkdir(likePath)
    local unlikePath = paths.concat(opt.output .. opt.threshold, 'unlike')
    paths.mkdir(unlikePath)
    
    local probability = opt.threshold
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
                image.save(paths.concat(likePath,
                    paths.basename(files[i + j - 1])) .. '.png', origins[i + j - 1])
            else
                image.save(paths.concat(unlikePath,
                    paths.basename(files[i + j - 1])) .. '.png', origins[i + j - 1])
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
            image.save(paths.concat(likePath,
                paths.basename(files[i + j - 1])) .. '.png', origins[i + j - 1])
        else
            image.save(paths.concat(unlikePath,
                paths.basename(files[i + j - 1])) .. '.png', origins[i + j - 1])
        end
    end
end

local function prepare(file)
    local inputImage = image.load(file, #channels)
    local originalImage = inputImage:clone()
    if inputImage:size()[2] ~= frameHeight and inputImage:size()[3] ~= frameWidth then
        inputImage = image.scale(inputImage, frameWidth, frameHeight)
    end
    
    images[#images + 1] = inputImage
    originalImages[#originalImages + 1] = originalImage
    fileNames[#fileNames + 1] = file
end


print(sys.COLORS.Yellow ..  '====> begin')
model = torch.load(opt.model)
if opt.type == 'cuda' then model = model:cuda() end
model:evaluate()
for file in paths.iterfiles(opt.input) do
    local f = paths.concat(opt.input, file)
    pcall(prepare ,f)
end
classify(images, originalImages, fileNames)

print(sys.COLORS._yellow ..  '====> end')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Classify beauty face')
cmd:text()
cmd:text('Options')
cmd:argument('-input','folder to classify')
cmd:option('-vgg', false, 'use vgg net')
cmd:option('-detectorThreshold', 0.9, 'face detector threshold')
cmd:option('-classifierThreshold', 0.8, 'face classifier threshold')
cmd:option('-devid', 1, 'cuda dev id')
cmd:text()
opt = cmd:parse(arg)

print('vgg:' .. tostring(opt.vgg) .. ' detectorThreshold:' .. opt.detectorThreshold
    .. ' classifierThreshold:' .. opt.classifierThreshold)

require 'detect'

local noFace = '_no_face'
local noBeautyFace = '_no_beauty_face'
local beautyFace = '_beauty_face'

print(sys.COLORS.Green ..  'processing: ' .. opt.input)

paths.mkdir(opt.input .. noFace)
paths.mkdir(opt.input .. noBeautyFace)
paths.mkdir(opt.input .. beautyFace)

for file in paths.iterfiles(opt.input) do
    local f = paths.concat(opt.input, file)
    local status, inputImage = pcall(image.load, f, 3)
    if status == false then
        print(sys.COLORS.red ..  '====> load image error ' .. f)
        goto continue
    end
    print(sys.COLORS.green ..  '====> processing file ' .. file)
    local beauty, img = pipeline(inputImage, opt.detectorThreshold, opt.classifierThreshold)
    if beauty == true then
        image.save(paths.concat(opt.input .. beautyFace, file), img)
    elseif img ~= nil then
        image.save(paths.concat(opt.input .. noBeautyFace, file), img)
    else
        image.save(paths.concat(opt.input .. noFace, file), inputImage)
    end
    ::continue::
end

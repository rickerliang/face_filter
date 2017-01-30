cmd = torch.CmdLine()
cmd:text()
cmd:text('Classify beauty face')
cmd:text()
cmd:text('Options')
cmd:argument('-input','file to classify')
cmd:option('-vgg', false, 'use vgg net')
cmd:option('-detectorThreshold', 0.9, 'face detector threshold')
cmd:option('-classifierThreshold', 0.9, 'face classifier threshold')
cmd:option('-devid', 1, 'cuda dev id')
cmd:text()
opt = cmd:parse(arg)

require 'detect'

local t = image.load(opt.input, 3)
local hasBeauty, ret = pipeline(t, opt.detectorThreshold, opt.classifierThreshold)
if ret ~= nil then
    image.save('output.png', ret)
end

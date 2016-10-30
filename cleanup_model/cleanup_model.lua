require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'

torch.setdefaulttensortype('torch.FloatTensor')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Clean up model')
cmd:text()
cmd:text('Options')
cmd:argument('-input','model find to clean up')
cmd:argument('-output', 'where to save')
cmd:text()
opt = cmd:parse(arg)

cutorch.setDevice(2)

local m = torch.load(opt.input)
m:clearState()
torch.save(opt.output, m)

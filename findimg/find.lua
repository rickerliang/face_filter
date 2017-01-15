require 'torch'
require 'paths'
require 'image'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Find exact the same image in a specific folder')
cmd:text()
cmd:text('Options')
cmd:argument('-input','pic to find')
cmd:argument('-folder', 'where to find')
cmd:text()
opt = cmd:parse(arg)

target = image.load(opt.input, 3)
for file in paths.iterfiles(opt.folder) do
    local f = paths.concat(opt.folder, file)
    img = image.load(f, 3)
    img = image.scale(img, target:size(3), target:size(2))
    diff = img:csub(target):abs():sum()
    if diff < 1000 then
        print(diff .. '-' .. f)
    end
end
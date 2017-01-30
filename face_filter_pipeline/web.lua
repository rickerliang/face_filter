local gm = require 'graphicsmagick'
local turbo = require 'turbo'
local uuid = require 'uuid'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
local ROOT = path.dirname(__FILE__)

local cmd = torch.CmdLine()
cmd:text()
cmd:text("")
cmd:text("Options:")
cmd:option('-vgg', false, 'use vgg net')
cmd:option("-port", 12121, 'listen port')
cmd:option('-detectorThreshold', 0.9, 'face detector threshold')
cmd:option('-classifierThreshold', 0.9, 'face classifier threshold')
cmd:option('-devid', 1, 'cuda dev id')
opt = cmd:parse(arg)

print('vgg:' .. tostring(opt.vgg) .. ' detectorThreshold:' .. opt.detectorThreshold
    .. ' classifierThreshold:' .. opt.classifierThreshold)

require 'detect'

local function getImage(req)
   local file_info = req:get_arguments("file")
   local url = req:get_argument("url", "")
   local file = nil
   local filename = nil
   if file_info and #file_info == 1 then
      file = file_info[1][1]
      local disp = file_info[1]["content-disposition"]
      if disp and disp["filename"] then
	  filename = path.basename(disp["filename"])
      end
   end
   if file and file:len() > 0 then
      local im = gm.Image()
      im:fromBlob(file, #file)
      local imageTensor = im:toTensor('float', 'RGB', 'DHW')
      return imageTensor
   end
   return nil
end

local DetectHandler = class("DetectHandler", turbo.web.RequestHandler)
function DetectHandler:post()
    local img = getImage(self)
    if img == nil then
        self:write("image decode fail")
        return
    end
    local hasBeauty, ret = pipeline(img, opt.detectorThreshold, opt.classifierThreshold)
    if hasBeauty == false and ret == nil then
        self:write("nothing detected")
        return
    end
    
    local detectedImg = gm.Image(ret, 'RGB', 'DHW')
    local name = uuid() .. ".png"
    local blob = detectedImg:format('PNG'):toString()
    self:set_header("Content-Length", string.format("%d", #blob))
    self:set_header("Content-Type", "image/png")
	self:set_header("Content-Disposition", string.format('inline; filename="%s"', name))
    self:write(blob)
end

local app = turbo.web.Application:new(
   {
      {"^/detect$", DetectHandler},
      {"^/([%a%d%.%-_]+)$", turbo.web.StaticFileHandler, path.join(ROOT, "assets/")},
   }
)
print("Listening 0.0.0.0:" .. opt.port)
app:listen(opt.port, "0.0.0.0", {max_body_size = CURL_MAX_SIZE})
turbo.ioloop.instance():start()

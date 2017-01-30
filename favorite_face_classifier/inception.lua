require 'cunn'
require 'cudnn'
require 'dpnn'

-- modify from dpnn.inception, change nn to cudnn
------------------------------------------------------------------------
-- [[ Inception ]]--
-- Uses n+2 parallel "columns". The original paper uses 2+2 where
-- the first two are (but there could be more than two):
-- 1x1 conv (reduce) -> relu -> 5x5 conv -> relu
-- 1x1 conv (reduce) -> relu -> 3x3 conv -> relu
-- and where the other two are :
-- 3x3 maxpool -> 1x1 conv (reduce/project) -> relu
-- 1x1 conv (reduce) -> relu.
-- This Model allows the first group of columns to be of any
-- number while the last group consist of exactly two columns.
-- The 1x1 conv are used to reduce the number of input channels
-- (or filters) such that the capacity of the network doesnt
-- explode. We refer to these here has "reduce". Since each
-- column seems to have one and only one reduce, their initial
-- configuration options are specified in lists of n+2 elements.
------------------------------------------------------------------------
local Inception, parent = torch.class("cudnn.Inception", "nn.Decorator")

function Inception:__init(config)
   --[[ Required Arguments ]]--
   -- Number of input channels or colors
   self.inputSize = config.inputSize
   -- Number of filters in the non-1x1 convolution kernel sizes, e.g. {32,48}
   self.outputSize = config.outputSize
   -- Number of filters in the 1x1 convolutions (reduction)
   -- used in each column, e.g. {48,64,32,32}. The last 2 are
   -- used respectively for the max pooling (projection) column
   -- (the last column in the paper) and the column that has
   -- nothing but a 1x1 conv (the first column in the paper).
   -- This table should have two elements more than the outputSize
   self.reduceSize = config.reduceSize

   --[[ Optional Arguments ]]--
   -- The strides of the 1x1 (reduction) convolutions. Defaults to {1,1,...}
   self.reduceStride = config.reduceStride or {}
   -- A transfer function like nn.Tanh, nn.Sigmoid, nn.ReLU, nn.Identity, etc.
   -- It is used after each reduction (1x1 convolution) and convolution
   self.transfer = config.transfer or nn.LeakyReLU(0.01)
   -- batch normalization can be awesome
   self.batchNorm = config.batchNorm
   -- Adding padding to the input of the convolutions such that
   -- input width and height are same as that of output.
   self.padding = true
   if config.padding ~= nil then
      self.padding = config.padding
   end
   -- The size (height=width) of the non-1x1 convolution kernels.
   self.kernelSize = config.kernelSize or {5,3}
   -- The stride (height=width) of the convolution.
   self.kernelStride = config.kernelStride or {1,1}
   -- The size (height=width) of the spatial max pooling used
   -- in the next-to-last column.
   self.poolSize = config.poolSize or 3
   -- The stride (height=width) of the spatial max pooling.
   self.poolStride = config.poolStride or 1
   -- The pooling layer.
   self.pool = config.pool or cudnn.SpatialMaxPooling(self.poolSize, self.poolSize, self.poolStride, self.poolStride)


   -- Variables checking that all of the output sizes are the same for a sample input.
   local iWidth, iHeight = 100, 200
   local oWidth, oHeight

   -- [[ Module Construction ]]--
   local depthConcat = nn.DepthConcat(2) -- concat on 'c' dimension
   -- 1x1 conv (reduce) -> 3x3 conv
   -- 1x1 conv (reduce) -> 5x5 conv
   -- ...
   for i=1,#self.kernelSize do
      local mlp = nn.Sequential()
      -- 1x1 conv
      local reduce = cudnn.SpatialConvolution(
         self.inputSize, self.reduceSize[i], 1, 1,
         self.reduceStride[i] or 1, self.reduceStride[i] or 1
      )
      mlp:add(reduce)
      if self.batchNorm then
         mlp:add(cudnn.SpatialBatchNormalization(self.reduceSize[i]))
      end
      mlp:add(self.transfer:clone())

      -- nxn conv
      local pad = self.padding and math.floor(self.kernelSize[i]/2) or 0
      local conv = cudnn.SpatialConvolution(
         self.reduceSize[i], self.outputSize[i],
         self.kernelSize[i], self.kernelSize[i],
         self.kernelStride[i], self.kernelStride[i],
         pad
      )
      mlp:add(conv)
      if self.batchNorm then
         mlp:add(cudnn.SpatialBatchNormalization(self.outputSize[i]))
      end
      mlp:add(self.transfer:clone())
      depthConcat:add(mlp)

      -- Check the output sizes.
      local oWidth_i = torch.floor(
         (iWidth + 2*pad - self.kernelSize[i])/self.kernelStride[i] + 1)
      local oHeight_i = torch.floor(
         (iHeight + 2*pad - self.kernelSize[i])/self.kernelStride[i] + 1)
      if oWidth == nil then
         oWidth = oWidth_i
         oHeight = oHeight_i
      else
         if oWidth ~= oWidth_i or oHeight ~= oHeight_i then
            print("inception: Warning: Inconsistent output sizes.")
         end
      end
   end

   -- pool -> 1x1 conv
   local mlp = nn.Sequential()
   mlp:add(self.pool)
   -- not sure if transfer should go here? mlp:add(transfer:clone())
   local i = #(self.kernelSize) + 1
   if self.reduceSize[i] then
      local reduce = cudnn.SpatialConvolution(
         self.inputSize, self.reduceSize[i], 1, 1,
         self.reduceStride[i] or 1, self.reduceStride[i] or 1
      )
      mlp:add(reduce)
      if self.batchNorm then
         mlp:add(cudnn.SpatialBatchNormalization(self.reduceSize[i]))
      end
      mlp:add(self.transfer:clone())
   end
   depthConcat:add(mlp)

   -- Check the output sizes. Infer the operation of the pooling layer.
   if self.pool.kW ~= nil and self.pool.dW ~= nil and self.pool.padW ~= nil then
      assert(oWidth ~= nil)
      assert(oHeight ~= nil)
      local oWidth_pool = torch.floor(
         (iWidth + 2*self.pool.padW - self.pool.kW)/self.pool.dW + 1)
      local oHeight_pool = torch.floor(
         (iHeight + 2*self.pool.padH - self.pool.kH)/self.pool.dH + 1)
      if oWidth ~= oWidth_pool or oHeight ~= oHeight_pool then
         print("inception: Warning: Inconsistent output sizes in pooling.")
      end
   end

   -- reduce: 1x1 conv (channel-wise pooling)
   i = i + 1
   if self.reduceSize[i] then
      local mlp = nn.Sequential()
      local reduce = cudnn.SpatialConvolution(
          self.inputSize, self.reduceSize[i], 1, 1,
          self.reduceStride[i] or 1, self.reduceStride[i] or 1
      )
      mlp:add(reduce)
      if self.batchNorm then
          mlp:add(cudnn.SpatialBatchNormalization(self.reduceSize[i]))
      end
      mlp:add(self.transfer:clone())
      depthConcat:add(mlp)

      -- Check the output sizes.
      local oWidth_conv = torch.floor((iWidth - 1)/(self.reduceStride[i] or 1) + 1)
      local oHeight_conv = torch.floor((iHeight - 1)/(self.reduceStride[i] or 1) + 1)
      if oWidth ~= oWidth_conv or oHeight ~= oHeight_conv then
         print("inception: Warning: Inconsistent output sizes in 1x1 conv.")
      end
   end

   parent.__init(self, depthConcat)
end

function Inception:updateOutput(input)
   local input = self:toBatch(input, 3)
   local output = self.module:updateOutput(input)
   self.output = self:fromBatch(output, 3)
   return self.output
end

function Inception:updateGradInput(input, gradOutput)
   local input, gradOutput = self:toBatch(input, 3), self:toBatch(gradOutput, 3)
   local gradInput = self.module:updateGradInput(input, gradOutput)
   self.gradInput = self:fromBatch(gradInput, 3)
   return self.gradInput
end

function Inception:accGradParameters(input, gradOutput, scale)
   local input, gradOutput = self:toBatch(input, 3), self:toBatch(gradOutput, 3)
   self.module:accGradParameters(input, gradOutput, scale)
end

function Inception:accUpdateGradParameters(input, gradOutput, lr)
   local input, gradOutput = self:toBatch(input, 3), self:toBatch(gradOutput, 3)
   self.module:accUpdateGradParameters(input, gradOutput, lr)
end


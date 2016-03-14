local SpatialUpSamplingZero, parent = torch.class('nn.SpatialUpSamplingZero', 'nn.Module')

--[[
Applies a 2D up-sampling over an input image composed of several input planes.

The upsampling is done using the simple nearest neighbor technique.

The Y and X dimensions are assumed to be the last 2 tensor dimensions.  For
instance, if the tensor is 4D, then dim 3 is the y dimension and dim 4 is the x.

owidth  = width*scale_factor
oheight  = height*scale_factor 
--]]

function SpatialUpSamplingZero:__init(scale)
   parent.__init(self)

   self.scale_factor = scale
   if self.scale_factor < 1 then
     error('scale_factor must be greater than 1')
   end
   if math.floor(self.scale_factor) ~= self.scale_factor then
     error('scale_factor must be integer')
   end
   self.inputSize = torch.LongStorage(4)
   self.outputSize = torch.LongStorage(4)
   self.usage = nil
end

function SpatialUpSamplingZero:updateOutput(input)
   if input:dim() ~= 4 and input:dim() ~= 3 then
     error('SpatialUpSamplingZero only support 3D or 4D tensors')
   end
   -- Copy the input size
   local xdim = input:dim()
   local ydim = input:dim() - 1
   for i = 1, input:dim() do
     self.inputSize[i] = input:size(i)
     self.outputSize[i] = input:size(i)
   end
   self.outputSize[ydim] = self.outputSize[ydim] + (self.outputSize[ydim] - 1) * (self.scale_factor - 1)
   self.outputSize[xdim] = self.outputSize[xdim] + (self.outputSize[xdim] - 1) * (self.scale_factor - 1)
   -- Resize the output if needed
   if input:dim() == 3 then
     self.output:resize(self.outputSize[1], self.outputSize[2], 
       self.outputSize[3]):zero()
   else
     self.output:resize(self.outputSize):zero()
   end
	print(self.output)
   input.THNN.SpatialUpSamplingZero_updateOutput(input:cdata(), self.output:cdata(), self.scale_factor)
   return self.output
end

function SpatialUpSamplingZero:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   input.THNN.SpatialUpSamplingZero_updateGradInput(gradOutput:cdata(), self.gradInput:cdata(), self.scale_factor)
   return self.gradInput
end

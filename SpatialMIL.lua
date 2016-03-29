require 'nn'

local SpatialMIL, parent = torch.class('nn.SpatialMIL', 'nn.Module')

function SpatialMIL:__init(mil_type)
    parent.__init(self) -- would inherit gradInput and output variables
    
    if mil_type then
        self.mil_type = mil_type
    else
        self.mil_type = 'milnor' -- noisy OR
    end
    
    self.mil_indices = torch.LongTensor()
end

function SpatialMIL:getMilTypeId()
    local mil_mode = 0
    
    if self.mil_type == 'milmax' then
        mil_mode = 1
    elseif self.mil_type == 'milnor' then
        mil_mode = 2
    elseif self.mil_type == 'milmaxnor' then
        mil_mode = 3
    end
    
    return mil_mode
end

function SpatialMIL:updateOutput(input)

    input.THNN.SpatialMIL_updateOutput(input:cdata(), self.output:cdata(), self.mil_indices:cdata(), self:getMilTypeId())

    return self.output
end

function SpatialMIL:updateGradInput(input, gradOutput)
    
    input.THNN.SpatialMIL_updateGradInput(input:cdata(), 
        self.output:cdata(), 
        gradOutput:cdata(), 
        self.gradInput:cdata(), 
        self:getMilTypeId())
    
    return self.gradInput
end

require 'nn'

local SpatialMIL, parent = torch.class('nn.SpatialMIL', 'nn.Module')

function SpatialMIL:__init(mil_type)
    parent.__init(self) -- would inherit gradInput and output variables
    
    if mil_type then
        self.mil_type = mil_type
    else
        self.mil_type = 'milnor' -- noisy OR
    end
    
    self.max_indices = nil
    
    self.width = 12
    self.height = 12
    self.tmp = torch.Tensor(self.width, self.height):fill(1)
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

    local timer = torch.Timer()
    input.THNN.SpatialMIL_updateOutput(input:cdata(), self.output:cdata(), self:getMilTypeId())
    print('Time: ', timer:time().real)
    
    local batch_size = input:size(1)
    local num_channels = input:size(2)
    assert(self.width == input:size(3))
    
    if self.mil_type == 'milmax' then
            
        local tmp = self.output:clone() 
        timer = torch.Timer()
        local max_concepts, max_indices = torch.max(input:view(batch_size, num_channels, -1), 3)
        max_indices = max_indices:squeeze(3) -- remove that 3rd dim
        max_concepts = max_concepts:squeeze(3)
        self.output:resizeAs(max_concepts):copy(max_concepts)
        self.max_indices = torch.type(max_concepts) == 'torch.CudaTensor' and torch.CudaTensor() or torch.LongTensor()
        self.max_indices:resizeAs(max_indices):copy(max_indices)
        print('Time: ', timer:time().real)
        
        print(' norm = ', torch.norm(self.output))
        print(' norm diff = ', torch.norm(tmp - self.output))
            
    elseif self.mil_type == 'milnor' then
        
        local tmp = self.output:clone() 
        
        timer = torch.Timer()
        for i=1, batch_size do
            for j=1, num_channels do
                local prob = 1
                input[i][j]:apply(function(x) prob = prob * (1 - x) end)
                -- local prob = torch.csub(self.tmp, input[i][j]):prod() -- slower
                self.output[i][j] = 1 - prob
            end 
        end
        print('Time: ', timer:time().real)
        
        print(' norm = ', torch.norm(self.output))
        print(' norm diff = ', torch.norm(tmp - self.output))
        
    elseif self.mil_type == 'milmaxnor' then
        
        local tmp = self.output:clone() 
        
        timer = torch.Timer()
        for i=1, batch_size do
            for j=1, num_channels do
                local prob = 1
                input[i][j]:apply(function(x) prob = prob * (1 - x) end)
                -- local prob = torch.csub(self.tmp, input[i][j]):prod() -- slower
                local max_prob = torch.max(input[i][j])
                self.output[i][j] = math.max(1 - prob, max_prob)
            end 
        end
        print('Time: ', timer:time().real)
        
        print(' norm = ', torch.norm(self.output))
        print(' norm diff = ', torch.norm(tmp - self.output))
        
    else
        error('Unknown MIL type', self.mil_type)
    end
        
    return self.output
end

function SpatialMIL:updateGradInput(input, gradOutput)
    
    local timer = torch.Timer()
    input.THNN.SpatialMIL_updateGradInput(input:cdata(), 
        self.output:cdata(), 
        gradOutput:cdata(), 
        self.gradInput:cdata(), 
        self:getMilTypeId())
    
    print('Time: ', timer:time().real)
    
    local batch_size = input:size(1)
    local num_channels = input:size(2)
    assert(self.width == input:size(3))
        
    if self.mil_type == 'milmax' then
        
        local tmp = self.gradInput:clone() 
        self.gradInput:zero()
        
        timer = torch.Timer()
        self.gradInput:resizeAs(input):zero()
        for i=1, batch_size do
            for j=1, num_channels do
                local max_idx = self.max_indices[i][j]
                local w_idx = math.ceil(max_idx/self.height)
                local h_idx = max_idx % self.height
                if h_idx == 0 then h_idx = self.height end
                self.gradInput[i][j][w_idx][h_idx] = gradOutput[i][j]
            end
        end
        print('Time: ', timer:time().real)
        print(' norm = ', torch.norm(self.gradInput))
        print(' norm diff = ', torch.norm(tmp - self.gradInput))
        
        
    elseif self.mil_type == 'milnor' then
        
        local tmp = self.gradInput:clone() 
        self.gradInput:zero()
        
        timer = torch.Timer()
        self.gradInput:resizeAs(input):fill(0)
        for i=1, batch_size do
            for j=1, num_channels do
                local p = torch.type(input) == 'torch.CudaTensor' and torch.CudaTensor() or torch.Tensor()
                p:resize(self.width, self.height):fill(1-self.output[i][j])
                local q = torch.csub(self.tmp, input[i][j])    
                self.gradInput[i][j] = p:cdiv(q):mul(gradOutput[i][j])
            end
        end
        print('Time: ', timer:time().real)
        
        print(' norm = ', torch.norm(self.gradInput))
        print(' norm diff = ', torch.norm(tmp - self.gradInput))
        
    elseif self.mil_type == 'milmaxnor' then
       
        local tmp = self.gradInput:clone() 
        
        timer = torch.Timer()
        self.gradInput:resizeAs(input):fill(1)
        for i=1, batch_size do
            for j=1, num_channels do
                local p = torch.type(input) == 'torch.CudaTensor' and torch.CudaTensor() or torch.Tensor()
                p:resize(self.width, self.height):fill(1-self.output[i][j])
                local q = torch.csub(self.tmp, input[i][j])    
                self.gradInput[i][j]:cmin(p:cdiv(q)):mul(gradOutput[i][j])
            end
        end
        print('Time: ', timer:time().real)
        print(' norm = ', torch.norm(self.gradInput))
        print(' norm diff = ', torch.norm(tmp - self.gradInput))
        
    else
        error('Unknown MIL type', sefl.mil_type)
    end

    return self.gradInput
end

local MultiLabelCrossEntropyCriterion, parent = torch.class('nn.MultiLabelCrossEntropyCriterion', 'nn.Criterion')

function MultiLabelCrossEntropyCriterion:__init(loss_weight, weights, sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
       self.sizeAverage = sizeAverage
    else
       self.sizeAverage = true
    end
    if weights then
       assert(weights:dim() == 1, "weights input should be 1-D Tensor")
       self.weights = weights
    end

    self.output_tensor = torch.zeros(1)
    self.total_weight_tensor = torch.ones(1)
    self.target = torch.zeros(1):long()
    
    -- support caffe loss weight
    if loss_weight then
        self.loss_weight = loss_weight
    else
        self.loss_weight = 1
    end
    assert(self.loss_weight > 0)
end

function MultiLabelCrossEntropyCriterion:__len()
    if (self.weights) then
        return #self.weights
    else
        return 0
    end
end

--[[
this implementation only penalizes correct labels
--]]

function MultiLabelCrossEntropyCriterion:updateOutput(input, target)
        
    -- check that input and target have the same shape
    assert(input:size(1) == target:size(1)) -- batch_size
    assert(input:size(2) == target:size(2)) -- num_target

    input.THNN.MultiLabelCrossEntropyCriterion_updateOutput(input:cdata(), target:cdata(), self.output_tensor:cdata())
    self.output = self.output_tensor[1]

    return self.output
end

-- Note that loss_weight is multipiled at element-wise
function MultiLabelCrossEntropyCriterion:updateGradInput(input, target)
            
    -- check that input and target have the same shape
    assert(input:size(1) == target:size(1)) -- batch_size
    assert(input:size(2) == target:size(2)) -- num_target

    input.THNN.MultiLabelCrossEntropyCriterion_updateGradInput(input:cdata(), target:cdata(), 
        self.gradInput:cdata(), self.loss_weight)
    
    return self.gradInput
end

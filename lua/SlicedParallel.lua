local SlicedParallel, parent = torch.class('nn.SlicedParallel', 'nn.Container')

-- slices is a table of sizes
function SlicedParallel:__init(slices, modules)
	parent.__init(self)
	self.slices = slices
	self.modules = modules
	self.moduleInputs = {}
	self.moduleOutputs = {}
end

function SlicedParallel:updateOutput(input)
	local slicingDim = input:dim()
	if input:dim() > 2 then
		error("Input must be vector or matrix")
	end

	local offset = 1
	for i, mod in ipairs(self.modules) do
		self.moduleOutputs[i] = mod:updateOutput(input:narrow(slicingDim, offset, self.slices[i]))
		offset = offset + self.slices[i]
	end

	self.output = torch.cat(self.moduleOutputs, slicingDim)
	return self.output
end

function SlicedParallel:updateGradInput(input, gradOutput)
	local slicingDim = input:dim()
	self.gradInput:resizeAs(input)

	local inputOffset = 1
	local outputOffset = 1
	for i, mod in ipairs(self.modules) do
		local moduleInput = input:narrow(slicingDim, inputOffset, self.slices[i])
		local moduleGradOutput = gradOutput:narrow(slicingDim, outputOffset, self.moduleOutputs[i]:size(slicingDim))
		self.gradInput:narrow(slicingDim, inputOffset, self.slices[i]):copy(mod:updateGradInput(moduleInput, moduleGradOutput))
		outputOffset = outputOffset + self.moduleOutputs[i]:size(slicingDim)
		inputOffset = inputOffset + self.slices[i]
	end

	return self.gradInput
end

function SlicedParallel:accGradParameters(input, gradOutput, scale)
	local slicingDim = input:dim()

	local inputOffset = 1
	local outputOffset = 1
	for i, mod in ipairs(self.modules) do
		local module_input = input:narrow(slicingDim, inputOffset, self.slices[i])
		local moduleGradOutput = gradOutput:narrow(slicingDim, outputOffset, self.moduleOutputs[i]:size(slicingDim))
		mod:accGradParameters(module_input, moduleGradOutput, scale)
		outputOffset = outputOffset + self.moduleOutputs[i]:size(slicingDim)
		inputOffset = inputOffset + self.slices[i]
	end
end

function SlicedParallel:accUpdateGradParameters(input, gradOutput, lr)
	local slicingDim = input:dim()

	local inputOffset = 1
	local outputOffset = 1
	for i, mod in ipairs(self.modules) do
		local module_input = input:narrow(slicingDim, inputOffset, self.slices[i])
		local moduleGradOutput = gradOutput:narrow(slicingDim, outputOffset, self.moduleOutputs[i]:size(slicingDim))
		mod:accUpdateGradParameters(module_input, moduleGradOutput, lr)
		outputOffset = outputOffset + self.moduleOutputs[i]:size(slicingDim)
		inputOffset = inputOffset + self.slices[i]
	end
end

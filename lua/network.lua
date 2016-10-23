package.path = package.path .. ';lua/?.lua'

require 'torch'
require 'nn'
require 'optim'

require 'SlicedParallel'

local input_tensor
local input_tensor_multiple
local net

local ELIGIBILITY_DECAY = 0.7

-- torch.setnumthreads(1)

local criterion = nn.MSECriterion():float()
local params, grad_params

-- for Adam
local optimizer_state_initial = { learningRate = 0.0003 }

local optimizer_state = optimizer_state_initial

local eligibility_trace

function init_parameters()
	params, grad_params = net:getParameters()
end

function init_eligibility_traces(batch_size)
	assert(params ~= nil)
	assert(batch_size ~= nil)
	eligibility_trace = torch.FloatTensor(batch_size, params:size(1)):zero()
end

function reset_eligibility_trace(batch)
	eligibility_trace:narrow(1, batch, 1):zero()
end

-- This implements the inner loop specified here in Figure 8.1:
-- https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node87.html
--
-- x_before is s
-- err is TD error
-- decay is lambda
-- gamma is hard-coded to be 1 at the moment
function update_with_eligibility_traces(x_before, err)
	-- First we compute V(s) and gradients of V(s), and update eligibility
	-- Since each backward requires a forward, it's more efficient to do these
	-- things at the same time.
	-- Unfortunately we need to do one element at a time, because Torch doesn't
	-- support batched gradient calculation.
	local one_tensor = torch.ones(1, 1):float()
	local batch_size = x_before:size(1)
	eligibility_trace:mul(ELIGIBILITY_DECAY)

	for i = 1, batch_size do
		grad_params:zero()
		local x = x_before:narrow(1, i, 1)
		net:forward(x)
		net:backward(x, one_tensor)

		-- Update eligibility trace
		eligibility_trace:narrow(1, i, 1):add(grad_params)
	end

	local err_expanded = torch.expand(err, batch_size, params:size(1)):clone()

	-- Now we have error and updated eligibility trace of all elements, we can update
	-- the model
	local function feval()
		return nil, torch.sum(err_expanded:cmul(eligibility_trace):mul(-1), 1)
	end

	_, _ = optim.adam(feval, params, optimizer_state)
end

function load(filename)
	net = torch.load(filename, 'ascii')
	init_parameters()
end

function save(filename)
	net:clearState()
	net.eigen = to_eigen_string()
	torch.save(filename, net, 'ascii')
end

function to_string()
	net:clearState()
	net.eigen = to_eigen_string()
	return torch.serialize(net, 'ascii')
end

function from_string(str)
	net = torch.deserialize(str, 'ascii')
	init_parameters()
end

function register_input_tensor(input)
	input_tensor = input
end

function register_input_tensor_multiple(input)
	input_tensor_multiple = input
end

-- This function is very latency-critical, so we don't want to have to pass the
-- input tensor every time. The caller should use register_input_tensor() to
-- set the input tensor instead, and just modify its value every time.
function forward_single()
	return net:forward(input_tensor)[1]
end

function forward_multiple()
	return net:forward(input_tensor_multiple)
end

function set_is_training(training)
	if training then
		net:training()
	else
		net:evaluate()
	end
end

function reset_optimizer()
	optimizer_state = optimizer_state_initial
end

function train_batch(x, t)
	local function feval(params)
		grad_params:zero()
		local y = net:forward(x)
		local loss = criterion:forward(y, t)
		net:backward(x, criterion:backward(y, t))
		return loss, grad_params
	end

	_, loss = optim.adam(feval, params, optimizer_state)
	return loss[1]
end

function make_evaluator(numInputs, slices, reductionFactors)
	local groupModules = {}
	assert(#slices == #reductionFactors)

	print("Slices:")
	print_table(slices)
	print("Reduction Factors:")
	print_table(reductionFactors)

	local totalNumInputs = 0
	local totalNumOutputs = 0
	for group = 1, #slices do
		local subMod = nn.Sequential()
		--subMod:add(nn.Linear(slices[group], slices[group]))
		--subMod:add(nn.ReLU())
		local output = math.ceil(slices[group] * reductionFactors[group])
		subMod:add(nn.Linear(slices[group], output))
		-- subMod:add(nn.PReLU(output))
		subMod:add(nn.ReLU())
		groupModules[group] = subMod
		totalNumInputs = totalNumInputs + slices[group]
		totalNumOutputs = totalNumOutputs + output
	end

	assert(totalNumInputs == numInputs)
	net = nn.Sequential()

	net:add(nn.SlicedParallel(slices, groupModules))

	--net:add(nn.BatchNormalization(totalNumOutputs))

	net:add(nn.Linear(totalNumOutputs, 64))
	--net:add(nn.PReLU(64))
	net:add(nn.ReLU())

	--net:add(nn.BatchNormalization(64))
	net:add(nn.Linear(64, 1))
	net:add(nn.Tanh())
	net:float()

	init_parameters()
end

function make_evaluator_heavy(numInputs, slices, reductionFactors)
	local groupModules = {}
	assert(#slices == #reductionFactors)

	print("Slices:")
	print_table(slices)
	print("Reduction Factors:")
	print_table(reductionFactors)

	local totalNumInputs = 0
	local totalNumOutputs = 0
	for group = 1, #slices do
		local subMod = nn.Sequential()
		--subMod:add(nn.Linear(slices[group], slices[group]))
		--subMod:add(nn.ReLU())
		local output = math.ceil(slices[group] * reductionFactors[group])
		subMod:add(nn.Linear(slices[group], output))
		subMod:add(nn.ReLU())
		groupModules[group] = subMod
		totalNumInputs = totalNumInputs + slices[group]
		totalNumOutputs = totalNumOutputs + output
	end

	assert(totalNumInputs == numInputs)
	net = nn.Sequential()

	net:add(nn.SlicedParallel(slices, groupModules))

	--net:add(nn.BatchNormalization(totalNumOutputs))

	net:add(nn.Linear(totalNumOutputs, 256))
	net:add(nn.ReLU())

	--net:add(nn.BatchNormalization(64))
	net:add(nn.Linear(256, 1))
	net:add(nn.Tanh())
	net:float()

	init_parameters()
end

function make_move_evaluator(numInputs)
	net = nn.Sequential()
	net:add(nn.Linear(numInputs, 256))
	net:add(nn.ReLU())
	net:add(nn.Dropout(0.5))
	net:add(nn.Linear(256, 64))
	net:add(nn.ReLU())
	net:add(nn.Dropout(0.5))
	net:add(nn.Linear(64, 1))
	net:add(nn.Tanh())
	net:float()

	init_parameters()
end

function tensor_to_string(tensor)
	local values = {}

	values[1] = tensor:dim()

	if tensor:dim() == 1 then
		values[2] = tensor:size(1)
		for i = 1, tensor:size(1) do
			values[#values + 1] = tensor[i]
		end
	elseif tensor:dim() == 2 then
		values[2] = tensor:size(1)
		values[3] = tensor:size(2)
		for i = 1, tensor:size(1) do
			for j = 1, tensor:size(2) do
				values[#values + 1] = tensor[i][j]
			end
		end
	else
		print("Tensor dim not supported")
	end

	return table.concat(values, ' ')
end

function to_eigen_string()
	local output = "EIGEN\n" .. mod_to_eigen_string(net) .. "\n"
	return output
end

function mod_to_eigen_string(mod)
	typename = torch.typename(mod)

	local output = ""

	if typename == "nn.Sequential" then
		output = output .. "nn.Sequential\n"
		output = output .. #mod.modules .. "\n"
		for _, sub_mod in pairs(mod.modules) do
			output = output .. mod_to_eigen_string(sub_mod) .. "\n"
		end
	elseif typename == "nn.SlicedParallel" then
		output = output .. "nn.SlicedParallel\n"
		output = output .. #mod.modules .. "\n"
		for part = 1, #mod.modules do
			output = output .. mod.slices[part] .. "\n"
			output = output .. mod_to_eigen_string(mod.modules[part]) .. "\n"
		end
	elseif typename == "nn.Linear" then
		output = output .. "nn.Linear\n"
		output = output .. tensor_to_string(mod.bias) .. "\n"
		output = output .. tensor_to_string(mod.weight) .. "\n"
	elseif typename == "nn.ReLU" then
		output = output .. "nn.ReLU\n"
	elseif typename == "nn.PReLU" then
		output = output .. "nn.PReLU\n"
		output = output .. tensor_to_string(mod.weight) .. "\n"
	elseif typename == "nn.Tanh" then
		output = output .. "nn.Tanh\n"
	elseif typename == "nn.Dropout" then
		output = output .. "nn.Dropout\n"
	elseif typename == "nn.BatchNormalization" then
		output = output .. "nn.BatchNormalization\n"
		output = output .. mod.eps .. "\n"
		output = output .. tensor_to_string(mod.running_mean) .. "\n"
		output = output .. tensor_to_string(mod.running_var) .. "\n"
		output = output .. tensor_to_string(mod.weight) .. "\n"
		output = output .. tensor_to_string(mod.bias) .. "\n"
	else
		print("Unknown typename " .. typename)
	end

	return output
end

function print_table(t)
	print("{")
	for key, value in pairs(t) do
		print("\t" .. key .. " = " .. value)
	end
	print("}")
end

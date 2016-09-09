require 'torch'
require 'nn'
require 'optim'

local input_tensor
local input_tensor_multiple
local net

-- torch.setnumthreads(1)

local criterion = nn.MSECriterion():float()
local params, grad_params

local optimizer_state = {}

function load(filename)
	net = torch.load(filename, 'ascii')
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

function train_batch(x, t)
	if params == nil then
		params, grad_params = net:getParameters()
	end

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

function make_evaluator(numInputs)
	net = nn.Sequential()
	net:add(nn.Linear(numInputs, 256))
	net:add(nn.ReLU())
	net:add(nn.Linear(256, 64))
	net:add(nn.ReLU())
	net:add(nn.Linear(64, 1))
	net:add(nn.Tanh())
	net:float()
end

function make_move_evaluator(numInputs)
	net = nn.Sequential()
	net:add(nn.Linear(numInputs, 256))
	net:add(nn.ReLU())
	net:add(nn.Linear(256, 64))
	net:add(nn.ReLU())
	net:add(nn.Linear(64, 1))
	net:add(nn.Tanh())
	net:float()
end

function make_simple(numInputs)
	net = nn.Sequential()
	net:add(nn.Linear(numInputs, 5))
	net:add(nn.ReLU())
	net:add(nn.Linear(5, 3))
	net:add(nn.Tanh())
	net:float()
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
	assert(torch.typename(net) == "nn.Sequential")
	local output = "EIGEN\n" .. #net.modules .. "\n"

	for _, mod in pairs(net.modules) do
		typename = torch.typename(mod)

		if typename == "nn.Linear" then
			output = output .. "nn.Linear\n"
			output = output .. tensor_to_string(mod.bias) .. "\n"
			output = output .. tensor_to_string(mod.weight) .. "\n"
		elseif typename == "nn.ReLU" then
			output = output .. "nn.ReLU\n"
		elseif typename == "nn.Tanh" then
			output = output .. "nn.Tanh\n"
		else
			print("Unknown typename " .. typename)
		end
	end

	return output
end

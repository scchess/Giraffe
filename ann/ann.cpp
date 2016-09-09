/*
	Copyright (C) 2015 Matthew Lai

	Giraffe is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Giraffe is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "ann.h"

#include "random_device.h"

ANN::ANN(bool eigenOnly)
	: m_eigenOnly(eigenOnly)
{
	if (!m_eigenOnly)
	{
		Init_();
	}
}

ANN::ANN(const std::string &networkFile)
{
	Init_();

	Load(networkFile);
}

ANN::ANN(const std::string &functionName, int numInputs)
{
	Init_();

	LuaFunctionCall<1, 0> makeCall(m_luaState, functionName.c_str());
	makeCall.PushInt(numInputs);
	makeCall.Call();
}

float ANN::Train(const NNMatrixRM &x, const NNMatrixRM &t)
{
	assert(!m_eigenOnly);

	if (!m_trainingX || !m_trainingT)
	{
		m_trainingX = THFloatTensor_newWithSize2d(x.rows(), x.cols());
		m_trainingT = THFloatTensor_newWithSize2d(t.rows(), t.cols());
		THFloatTensor_retain(m_trainingX);
		THFloatTensor_retain(m_trainingT);
	}

	m_eigenAnnUpToDate = false;

	assert(x.rows() == THFloatTensor_size(m_trainingX, 0));
	assert(x.cols() == THFloatTensor_size(m_trainingX, 1));
	assert(t.rows() == THFloatTensor_size(m_trainingT, 0));
	assert(t.cols() == 1);

	Eigen::Map<NNMatrixRM> xMap(THFloatTensor_data(m_trainingX), x.rows(), x.cols());
	Eigen::Map<NNMatrixRM> tMap(THFloatTensor_data(m_trainingT), t.rows(), t.cols());
	xMap = x;
	tMap = t;

	LuaFunctionCall<2, 1> trainCall(m_luaState, "train_batch");
	trainCall.PushTensor(m_trainingX);
	trainCall.PushTensor(m_trainingT);
	trainCall.Call();
	return trainCall.PopNumber(); // loss
}

void ANN::Load(const std::string &filename)
{
	assert(!m_eigenOnly);
	LuaFunctionCall<1, 0> loadCall(m_luaState, "load");
	loadCall.PushString(filename);
	loadCall.Call();

	m_eigenAnn.FromString(ToString());
	m_eigenAnnUpToDate = true;
}

void ANN::Save(const std::string &filename)
{
	assert(!m_eigenOnly);
	LuaFunctionCall<1, 0> loadCall(m_luaState, "save");
	loadCall.PushString(filename);
	loadCall.Call();
}

std::string ANN::ToString() const
{
	assert(!m_eigenOnly);
	std::lock_guard<std::mutex> lock(m_mutex);
	LuaFunctionCall<0, 1> loadCall(m_luaState, "to_string");
	loadCall.Call();
	std::string s = loadCall.PopString();
	return s;
}

void ANN::FromString(const std::string &s)
{
	if (!m_eigenOnly)
	{
		LuaFunctionCall<1, 0> loadCall(m_luaState, "from_string");
		loadCall.PushString(s);
		loadCall.Call();
	}

	m_eigenAnn.FromString(s);
	m_eigenAnnUpToDate = true;
}

ANN::~ANN()
{
	auto checkFreeTensorFcn = [](THFloatTensor *&tensor)
	{
		if (tensor != nullptr)
		{
			THFloatTensor_free(tensor);
			tensor = nullptr;
		}
	};

	checkFreeTensorFcn(m_inputTensorSingle);
	checkFreeTensorFcn(m_inputTensorMultiple);
	checkFreeTensorFcn(m_trainingX);
	checkFreeTensorFcn(m_trainingT);

	if (m_luaState != nullptr)
	{
		lua_close(m_luaState);
		m_luaState = nullptr;
	}
}

void ANN::Init_()
{
	m_luaState = luaL_newstate();
	luaL_openlibs(m_luaState);
	if (luaL_dofile(m_luaState, "lua/network.lua") != 0)
	{
		std::cerr << "Failed to load lua/network.lua: " << lua_tostring(m_luaState, -1) << std::endl;
		assert(false);
	}
}

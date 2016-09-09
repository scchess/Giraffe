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

#ifndef ANN_H
#define ANN_H

#include <random>
#include <string>
#include <ostream>
#include <istream>
#include <iostream>
#include <mutex>

#include <cmath>
#include <cassert>

#include <omp.h>

#include <lua.hpp>
#include <luaT.h>
#include <lauxlib.h>
#include <TH/THTensor.h>

#include "eigen_ann.h"
#include "matrix_ops.h"

namespace
{
template<int NUM_ARGUMENTS, int NUM_RETS>
class LuaFunctionCall
{
public:
	LuaFunctionCall(lua_State *state, const char *name)
		: m_state(state), m_name(name)
	{
		lua_getglobal(m_state, name);
	}

	void PushString(const std::string &str)
	{
		lua_pushstring(m_state, str.c_str());
		++m_numArgsPushed;
	}

	void PushInt(int x)
	{
		lua_pushinteger(m_state, x);
		++m_numArgsPushed;
	}

	void PushTensor(THFloatTensor *tensor)
	{
		THFloatTensor_retain(tensor);
		luaT_pushudata(m_state, (void *) tensor, "torch.FloatTensor");
		++m_numArgsPushed;
	}

	float PopNumber()
	{
		float ret = lua_tonumber(m_state, -1);
		lua_remove(m_state, -1);
		++m_numRetsPopped;
		return ret;
	}

	THFloatTensor* PopTensor()
	{
		THFloatTensor *ret = reinterpret_cast<THFloatTensor*>(luaT_toudata(m_state, -1, "torch.FloatTensor"));
		lua_remove(m_state, -1);
		++m_numRetsPopped;
		return ret;
	}

	std::string PopString()
	{
		const char *p = lua_tostring(m_state, -1);
		std::string ret = p;
		lua_remove(m_state, -1);
		++m_numRetsPopped;
		return ret;
	}

	void Call()
	{
		assert(m_numArgsPushed == NUM_ARGUMENTS);
		if (lua_pcall(m_state, NUM_ARGUMENTS, NUM_RETS, 0) != 0)
		{
			std::cerr << "Lua function call " << m_name << " failed: " << lua_tostring(m_state, -1) << std::endl;
			assert(false);
		}
	}

	~LuaFunctionCall()
	{
		if (m_numRetsPopped != NUM_RETS)
		{
			std::cerr << "Wrong number of return values popped! " << m_name << std::endl;
			assert(false);
		}
	}

	LuaFunctionCall(const LuaFunctionCall &) = delete;
	LuaFunctionCall &operator=(const LuaFunctionCall &) = delete;

private:
	lua_State *m_state;
	const char *m_name;
	int m_numArgsPushed = 0;
	int m_numRetsPopped = 0;
};
}

class ANN
{
public:
	ANN(bool eigenOnly = false);
	ANN(const std::string &networkFile);
	ANN(const std::string &functionName, int numInputs);

	template <typename Derived>
	float ForwardSingle(const Eigen::MatrixBase<Derived> &v);

	template <typename Derived>
	NNMatrixRM ForwardMultiple(const Eigen::MatrixBase<Derived> &x);

	float Train(const NNMatrixRM &x, const NNMatrixRM &t);

	void Load(const std::string &filename);
	void Save(const std::string &filename);

	std::string ToString() const;
	void FromString(const std::string &s);

	ANN &operator=(ANN &&other)
	{
		m_luaState = other.m_luaState;
		m_inputTensorSingle = other.m_inputTensorSingle;
		m_inputTensorMultiple = other.m_inputTensorMultiple;
		m_trainingX = other.m_trainingX;
		m_trainingT = other.m_trainingT;
		m_eigenAnnUpToDate = false;

		other.m_luaState = nullptr;
		other.m_inputTensorSingle = nullptr;
		other.m_inputTensorMultiple = nullptr;
		other.m_trainingX = nullptr;
		other.m_trainingT = nullptr;

		return *this;
	}

	ANN(ANN &&other)
	{
		*this = std::move(other);
	}

	~ANN();

private:
	void Init_();

	bool m_eigenOnly = false;
	bool m_eigenAnnUpToDate = false;
	EigenANN m_eigenAnn;

	mutable lua_State *m_luaState;

	THFloatTensor *m_inputTensorSingle = nullptr;
	THFloatTensor *m_inputTensorMultiple = nullptr;
	THFloatTensor *m_trainingX = nullptr;
	THFloatTensor *m_trainingT = nullptr;

	// This is for ToString(), which may be called from multiple threads
	mutable std::mutex m_mutex;
};

template <typename Derived>
float ANN::ForwardSingle(const Eigen::MatrixBase<Derived> &v)
{
#if 1
	if (!m_eigenAnnUpToDate)
	{
		m_eigenAnn.FromString(ToString());
		m_eigenAnnUpToDate = true;
	}

	return m_eigenAnn.ForwardSingle(v);
#else
	if (!m_inputTensorSingle)
	{
		m_inputTensorSingle = THFloatTensor_newWithSize1d(v.size());
		THFloatTensor_retain(m_inputTensorSingle);

		LuaFunctionCall<1, 0> registerCall(m_luaState, "register_input_tensor");
		registerCall.PushTensor(m_inputTensorSingle);
		registerCall.Call();
	}

	Eigen::Map<NNVector> tensorMap(THFloatTensor_data(m_inputTensorSingle), v.size());
	tensorMap = v;

	LuaFunctionCall<0, 1> forwardCall(m_luaState, "forward_single");
	forwardCall.Call();
	float torchOutput = forwardCall.PopNumber();
	return torchOutput;
#endif
}

template <typename Derived>
NNMatrixRM ANN::ForwardMultiple(const Eigen::MatrixBase<Derived> &x)
{
	if (!m_inputTensorMultiple)
	{
		m_inputTensorMultiple = THFloatTensor_newWithSize2d(x.rows(), x.cols());
		THFloatTensor_retain(m_inputTensorMultiple);

		LuaFunctionCall<1, 0> registerCall(m_luaState, "register_input_tensor_multiple");
		registerCall.PushTensor(m_inputTensorMultiple);
		registerCall.Call();
	}

	Eigen::Map<NNMatrixRM> tensorMap(THFloatTensor_data(m_inputTensorMultiple), x.rows(), x.cols());
	tensorMap = x;

	LuaFunctionCall<0, 1> forwardCall(m_luaState, "forward_multiple");
	forwardCall.Call();
	auto tensor = forwardCall.PopTensor();
	Eigen::Map<NNMatrixRM> returnedTensorMap(THFloatTensor_data(tensor), x.rows(), 1);

	NNMatrixRM ret(returnedTensorMap);
	return ret;
}

#endif // ANN_H

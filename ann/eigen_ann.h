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

#ifndef EIGEN_ANN_H
#define EIGEN_ANN_H

#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>

#include <Eigen/Core>

#include "matrix_ops.h"

class Layer
{
public:
	virtual NNMatrixRM *Forward(NNMatrixRM *input) = 0;
	virtual NNVector *ForwardSingle(NNVector *input) = 0;
};

// This is an alternative implementation of ANN that uses Eigen and doesn't
// rely on Lua/Torch. It only supports forward, but is much faster in gameplay.
class EigenANN
{
public:
	EigenANN() {}
	EigenANN(const std::string &filename)
	{
		Load(filename);
	}

	EigenANN(const std::string &/*functionName*/, int /*numInputs*/) { throw std::logic_error("Not supported"); }

	template <typename Derived>
	float ForwardSingle(const Eigen::MatrixBase<Derived> &v)
	{
		m_inputSingle = v;
		NNVector *x = &m_inputSingle;

		for (auto &layer : m_layers)
		{
			x = layer->ForwardSingle(x);
		}

		return (*x)(0, 0);
	}

	template <typename Derived>
	NNMatrixRM ForwardMultiple(const Eigen::MatrixBase<Derived> &matrix)
	{
		m_input = matrix;
		NNMatrixRM *x = &m_input;

		for (auto &layer : m_layers)
		{
			x = layer->Forward(x);
		}

		return *x;
	}

	float Train(const NNMatrixRM &/*x*/, const NNMatrixRM &/*t*/) { throw std::logic_error("Not supported"); }

	void Load(const std::string &filename)
	{
		std::ifstream infile(filename);
		std::stringstream ss;
		ss << infile.rdbuf();
		FromString(ss.str());
	}

	void Save(const std::string &/*filename*/) { throw std::logic_error("Not supported"); }

	std::string ToString() const { return m_stringRep; }
	void FromString(const std::string &s);

private:
	std::vector<std::unique_ptr<Layer>> m_layers;
	std::string m_stringRep;

	NNVector m_inputSingle;
	NNMatrixRM m_input;
};

class LinearLayer : public Layer
{
public:
	LinearLayer(const NNVector &bias, const NNMatrixRM &weight)
		: m_bias(bias), m_weight(weight) {}

	NNMatrixRM *Forward(NNMatrixRM *input) override {
		m_output.noalias() = *input * m_weight.transpose() + m_bias;
		return &m_output;
	}

	NNVector *ForwardSingle(NNVector *input) override {
		m_outputSingle.noalias() = *input * m_weight.transpose() + m_bias;
		return &m_outputSingle;
	}

private:
	NNVector m_bias;
	NNMatrixRM m_weight;

	NNMatrixRM m_output;
	NNVector m_outputSingle;
};

class ReLULayer : public Layer
{
public:
	NNMatrixRM *Forward(NNMatrixRM *input) override {
		*input = input->array().max(NNMatrixRM::Zero(input->rows(), input->cols()).array());
		return input;
	}

	NNVector *ForwardSingle(NNVector *input) override {
		*input = input->array().max(NNVector::Zero(input->rows(), input->cols()).array());
		return input;
	}
};

class TanhLayer : public Layer
{
public:
	NNMatrixRM *Forward(NNMatrixRM *input) override {
		// This is not vectorized unfortunately, but we should only be doing 1 per forward!
		for (int64_t i = 0; i < input->rows(); ++i)
		{
			for (int64_t j = 0; j < input->cols(); ++j)
			{
				(*input)(i, j) = std::tanh((*input)(i, j));
			}
		}
		return input;
	}

	NNVector *ForwardSingle(NNVector *input) override {
		for (int64_t i = 0; i < input->size(); ++i)
		{
			(*input)(i) = std::tanh((*input)(i));
		}
		return input;
	}
};

#endif // EIGEN_ANN_H

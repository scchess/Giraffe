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

class Module
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
	EigenANN(bool /*eigenOnly*/ = false) {}
	EigenANN(const std::string &filename)
	{
		Load(filename);
	}

	EigenANN(const std::string &/*functionName*/, int /*numInputs*/) {}
	EigenANN(const std::string &/*functionName*/, int /*numInputs*/, 
			 const std::vector<int64_t> &/*slices*/, const std::vector<float> &/*reductionFactors*/)
	{}

	template <typename Derived>
	float ForwardSingle(const Eigen::MatrixBase<Derived> &v)
	{
		m_inputSingle = v;
		NNVector *ret = m_module->ForwardSingle(&m_inputSingle);

		return (*ret)(0, 0);
	}

	template <typename Derived>
	NNMatrixRM *ForwardMultiple(const Eigen::MatrixBase<Derived> &x, bool useTorch = false)
	{
		assert(!useTorch);
		m_input = x;
		return m_module->Forward(&m_input);
	}

	float Train(const NNMatrixRM &/*x*/, const NNMatrixRM &/*t*/) { throw std::logic_error("Not supported"); }

	void ResetOptimizer() { throw std::logic_error("Not supported"); }

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
	std::unique_ptr<Module> m_module;
	std::string m_stringRep;

	NNVector m_inputSingle;
	NNMatrixRM m_input;
};

std::unique_ptr<Module> ReadModule(std::istream &is);

class Sequential : public Module
{
public:
	NNMatrixRM *Forward(NNMatrixRM *input) override {
		NNMatrixRM *x = input;
		for (auto &module : m_modules)
		{
			x = module->Forward(x);
		}

		return x;
	}

	NNVector *ForwardSingle(NNVector *input) override {
		NNVector *v = input;
		for (auto &module : m_modules)
		{
			v = module->ForwardSingle(v);
		}

		return v;
	}

	void AddModule(std::unique_ptr<Module> &&mod)
	{
		m_modules.push_back(std::move(mod));
	}

private:
	std::vector<std::unique_ptr<Module>> m_modules;
};

class SlicedParallel : public Module
{
public:
	NNMatrixRM *Forward(NNMatrixRM *input) override {
		int64_t offset = 0;
		for (size_t i = 0; i < m_modules.size(); ++i)
		{
			m_slices[i] = input->middleCols(m_sliceIndices[i], m_sliceSizes[i]);
			NNMatrixRM *modOutput = m_modules[i]->Forward(&m_slices[i]);

			m_output.resize(modOutput->rows(), Eigen::NoChange);

			if (m_output.cols() < (offset + modOutput->cols()))
			{
				m_output.conservativeResize(Eigen::NoChange, offset + modOutput->cols());
			}

			m_output.middleCols(offset, modOutput->cols()) = *modOutput;
			offset += modOutput->cols();
		}

		return &m_output;
	}

	NNVector *ForwardSingle(NNVector *input) override {
		int64_t offset = 0;
		for (size_t i = 0; i < m_modules.size(); ++i)
		{
			m_slicesSingle[i] = input->middleCols(m_sliceIndices[i], m_sliceSizes[i]);
			NNVector *modOutput = m_modules[i]->ForwardSingle(&m_slicesSingle[i]);

			if (m_outputSingle.cols() < (offset + modOutput->cols()))
			{
				m_outputSingle.conservativeResize(Eigen::NoChange, offset + modOutput->cols());
			}

			m_outputSingle.middleCols(offset, modOutput->cols()) = *modOutput;
			offset += modOutput->cols();
		}

		return &m_outputSingle;
	}

	void AddModule(std::unique_ptr<Module> &&mod, int64_t sliceIndex, int64_t sliceSize)
	{
		m_modules.push_back(std::move(mod));
		m_sliceIndices.push_back(sliceIndex);
		m_sliceSizes.push_back(sliceSize);
		m_slices.resize(m_sliceIndices.size());
		m_slicesSingle.resize(m_sliceIndices.size());
	}

private:
	std::vector<std::unique_ptr<Module>> m_modules;
	std::vector<int64_t> m_sliceIndices;
	std::vector<int64_t> m_sliceSizes;

	std::vector<NNMatrixRM> m_slices;
	std::vector<NNVector> m_slicesSingle;

	NNMatrixRM m_output;
	NNVector m_outputSingle;
};

class LinearLayer : public Module
{
public:
	LinearLayer(const NNVector &bias, const NNMatrixRM &weight)
		: m_bias(bias), m_weight(weight) {}

	NNMatrixRM *Forward(NNMatrixRM *input) override {
		m_output.noalias() = *input * m_weight.transpose();
		m_output.rowwise() += m_bias;
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

class DropoutLayer : public Module
{
public:
	// DropOut is no-op at evaluation time (output is scaled during training).
	NNMatrixRM *Forward(NNMatrixRM *input) override {
		return input;
	}

	NNVector *ForwardSingle(NNVector *input) override {
		return input;
	}
};

class ReLULayer : public Module
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

class PReLULayer : public Module
{
public:
	PReLULayer(const NNVector &weight)
		: m_weight(weight) {}

	NNMatrixRM *Forward(NNMatrixRM *input) override {
		*input =
			input->array().max(NNMatrixRM::Zero(input->rows(), input->cols()).array()) +
			input->array().min(NNMatrixRM::Zero(input->rows(), input->cols()).array() * m_weight.array());
		return input;
	}

	NNVector *ForwardSingle(NNVector *input) override {
		*input = 
			input->array().max(NNVector::Zero(input->rows(), input->cols()).array()) +
			input->array().min(NNVector::Zero(input->rows(), input->cols()).array() * m_weight.array());
		return input;
	}

private:
	NNVector m_weight;
};

class TanhLayer : public Module
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

class BatchNormLayer : public Module
{
public:
	BatchNormLayer(float eps, const NNVector &mean, const NNVector &var, const NNVector &weight, const NNVector &bias) {
		// This module computes y = ((x - mean) / sqrt(var + eps) * weight + bias
		// That's equivalent to y = (w/sqrt(var + eps)) * x - w*mean/sqrt(var + eps) + bias
		// or y = ax + b, where a = (w/sqrt(var + eps)), b = bias - w*mean/sqrt(var + eps)

		m_a = weight.array() / (var.array() + eps).sqrt();
		m_b = bias.array() - weight.array() * mean.array() / (var.array() + eps).sqrt();
	}

	NNMatrixRM *Forward(NNMatrixRM *input) override {
		for (int row = 0; row < input->rows(); ++row)
		{
			input->row(row).array() *= m_a.array();
		}
		input->rowwise() += m_b;
		return input;
	}

	NNVector *ForwardSingle(NNVector *input) override {
		input->array() *= m_a.array();
		input->array() += m_b.array();
		return input;
	}

private:
	NNVector m_a;
	NNVector m_b;
};

#endif // EIGEN_ANN_H

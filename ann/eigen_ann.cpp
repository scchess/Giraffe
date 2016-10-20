/*
	Copyright (C) 2016 Matthew Lai

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

#include "eigen_ann.h"

namespace
{

NNVector ReadVector(std::istream &is)
{
	int numDims;
	is >> numDims;
	assert(numDims == 1);
	int64_t elements;
	is >> elements;
	NNVector ret(elements);

	for (int64_t i = 0; i < elements; ++i)
	{
		is >> ret(i);
	}

	return ret;
}

NNMatrixRM ReadMatrix(std::istream &is)
{
	int numDims;
	is >> numDims;
	assert(numDims == 2);
	int64_t rows;
	is >> rows;
	int64_t cols;
	is >> cols;
	NNMatrixRM ret(rows, cols);

	for (int64_t row = 0; row < rows; ++row)
	{
		for (int64_t col = 0; col < cols; ++col)
		{
			is >> ret(row, col);
		}
	}

	return ret;
}

}

void EigenANN::FromString(const std::string &s)
{
	m_stringRep = s;

	std::stringstream ss(s);
	std::string token;

	// First we have to find the "EIGEN" marker
	do
	{
		ss >> token;
	}
	while (token != "EIGEN");

	m_module = ReadModule(ss);
}

std::unique_ptr<Module> ReadModule(std::istream &is)
{
	std::string layerType;
	is >> layerType;

	if (layerType == "nn.Sequential")
	{
		int num_modules;
		is >> num_modules;
		std::unique_ptr<Sequential> seq(new Sequential);

		for (int i = 0; i < num_modules; ++i)
		{
			seq->AddModule(ReadModule(is));
		}

		return seq;
	}
	else if (layerType == "nn.SlicedParallel")
	{
		int num_modules;
		is >> num_modules;
		std::unique_ptr<SlicedParallel> par(new SlicedParallel);

		int64_t offset = 0;
		for (int i = 0; i < num_modules; ++i)
		{
			int64_t size;
			is >> size;
			par->AddModule(ReadModule(is), offset, size);
			offset += size;
		}

		return par;
	}
	else if (layerType == "nn.Linear")
	{
		NNVector bias = ReadVector(is);
		NNMatrix weight = ReadMatrix(is);

		return std::unique_ptr<Module>(new LinearLayer(bias, weight));
	}
	else if (layerType == "nn.ReLU")
	{
		return std::unique_ptr<Module>(new ReLULayer);
	}
	else if (layerType == "nn.PReLU")
	{
		NNVector weight = ReadVector(is);
		return std::unique_ptr<Module>(new PReLULayer(weight));
	}
	else if (layerType == "nn.Tanh")
	{
		return std::unique_ptr<Module>(new TanhLayer);
	}
	else if (layerType == "nn.Dropout")
	{
		return std::unique_ptr<Module>(new DropoutLayer);
	}
	else if (layerType == "nn.BatchNormalization")
	{
		float eps;
		NNVector mean;
		NNVector var;
		NNVector weight;
		NNVector bias;

		is >> eps;
		mean = ReadVector(is);
		var = ReadVector(is);
		weight = ReadVector(is);
		bias = ReadVector(is);

		return std::unique_ptr<Module>(new BatchNormLayer(eps, mean, var, weight, bias));
	}
	else
	{
		std::cerr << "Layer type " << layerType << " not implemented!" << std::endl;
		assert(false);
	}

	return std::unique_ptr<Module>();
}

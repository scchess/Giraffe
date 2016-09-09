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

#include "ann_evaluator.h"

#include <fstream>
#include <set>

#include "consts.h"

ANNEvaluator::ANNEvaluator(bool eigenOnly)
	: m_ann(eigenOnly), m_evalHash(EvalHashSize)
{
	InvalidateCache();
}

ANNEvaluator::ANNEvaluator(const std::string &filename)
	: m_evalHash(EvalHashSize)
{
	Deserialize(filename);
}

void ANNEvaluator::BuildANN(int64_t inputDims)
{
	m_ann = ANN("make_evaluator", inputDims);
}

void ANNEvaluator::Serialize(const std::string &filename)
{
	m_ann.Save(filename);
}

void ANNEvaluator::Deserialize(const std::string &filename)
{
	m_ann.Load(filename);

	InvalidateCache();
}

float ANNEvaluator::Train(const NNMatrixRM &x, const NNMatrixRM &t)
{
	InvalidateCache();

	return m_ann.Train(x, t);
}

Score ANNEvaluator::EvaluateForWhiteImpl(Board &b, Score /*lowerBound*/, Score /*upperBound*/)
{
	auto hashResult = HashProbe_(b);

	if (hashResult)
	{
		return *hashResult;
	}

	FeaturesConv::ConvertBoardToNN(b, m_convTmp);

	// we have to map every time because the vector's buffer could have moved
	Eigen::Map<NNVector> mappedVec(&m_convTmp[0], 1, m_convTmp.size());

	float annOut = m_ann.ForwardSingle(mappedVec);

	Score nnRet = annOut * EvalFullScale;

	HashStore_(b, nnRet);

	return nnRet;
}

void ANNEvaluator::PrintDiag(Board &board)
{
	FeaturesConv::ConvertBoardToNN(board, m_convTmp);

	Eigen::Map<NNVector> mappedVec(&m_convTmp[0], 1, m_convTmp.size());

	std::cout << "Val: " << m_ann.ForwardSingle(mappedVec) << std::endl;
}

void ANNEvaluator::InvalidateCache()
{
	for (auto &entry : m_evalHash)
	{
		entry.hash = 0;
	}
}

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
#include <sstream>

#include "consts.h"
#include "util.h"

namespace
{

std::string ensembleFilename(const std::string &base, size_t num)
{
	std::stringstream ss;
	ss << base << "." << num;
	return ss.str();
}


}

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

void ANNEvaluator::BuildEnsemble()
{
	m_ensemble.clear();

	auto numFeatures = FeaturesConv::GetNumFeatures();
	auto featuresGroups = FeaturesConv::GetBoardGroupAllocations();
	std::vector<int64_t> slices;
	std::vector<float> reductionFactors;
	for (auto &group : featuresGroups)
	{
		slices.push_back(group.first);
		reductionFactors.push_back(group.second);
	}

	for (size_t i = 0; i < EnsembleSize; ++i)
	{
		m_ensemble.emplace_back("make_evaluator_heavy", numFeatures, slices, reductionFactors);
	}
}

void ANNEvaluator::LoadEnsemble(const std::string &baseFilename)
{
	size_t i = 0;
	m_ensemble.clear();

	while (true)
	{
		std::string filename = ensembleFilename(baseFilename, i);

		if (FileReadable(filename))
		{
			ANN ann(filename);
			m_ensemble.push_back(std::move(ann));
		}
	}

	std::cout << m_ensemble.size() << " nets in ensemble read" << std::endl;
}

void ANNEvaluator::SaveEnsemble(const std::string &baseFilename)
{
	for (size_t i = 0; i < m_ensemble.size(); ++i)
	{
		std::string filename = ensembleFilename(baseFilename, i);
		m_ensemble[i].Save(filename);
	}

	std::cout << m_ensemble.size() << " nets in ensemble saved" << std::endl;
}

void ANNEvaluator::BuildANN()
{
	auto numFeatures = FeaturesConv::GetNumFeatures();
	auto featuresGroups = FeaturesConv::GetBoardGroupAllocations();
	std::vector<int64_t> slices;
	std::vector<float> reductionFactors;
	for (auto &group : featuresGroups)
	{
		slices.push_back(group.first);
		reductionFactors.push_back(group.second);
	}
	m_ann = ANN("make_evaluator", numFeatures, slices, reductionFactors);
}

void ANNEvaluator::Serialize(const std::string &filename)
{
	m_ann.Save(filename);

	InvalidateCache();
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

float ANNEvaluator::TrainWithEnsemble(const NNMatrixRM &x, const NNMatrixRM &t)
{
	InvalidateCache();

	NNMatrixRM sumPredictions = NNMatrixRM::Zero(t.rows(), t.cols());

	for (size_t i = 0; i < m_ensemble.size(); ++i)
	{
		m_ensemble[i].Train(x, t);

		// run forward again to get new output
		NNMatrixRM *output = m_ensemble[i].ForwardMultiple(x, true /* use torch */);
		sumPredictions += *output;
	}

	return m_ann.Train(x, (sumPredictions.array() / static_cast<float>(m_ensemble.size())).matrix());
}

NNMatrixRM *ANNEvaluator::EvaluateMatrixWTM(const NNMatrixRM x)
{
	return m_ann.ForwardMultiple(x);
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

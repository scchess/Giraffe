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

ANNEvaluator::ANNEvaluator()
	: m_evalHash(EvalHashSize)
{
	InvalidateCache();
}

ANNEvaluator::ANNEvaluator(const std::string &filename)
	: m_evalHash(EvalHashSize)
{
	std::ifstream netfIn(filename);
	Deserialize(netfIn);
}

void ANNEvaluator::BuildANN(int64_t inputDims)
{
	m_ann = AnnBuilder::BuildEvalNet(inputDims, 1, false);
}

void ANNEvaluator::Serialize(std::ostream &os)
{
	SerializeNet(m_ann, os);
}

void ANNEvaluator::Deserialize(std::istream &is)
{
	DeserializeNet(m_ann, is);

	InvalidateCache();
}

float ANNEvaluator::Train(const NNMatrixRM &pred, EvalNet::Activations &act, const NNMatrixRM &targets)
{
	NNMatrixRM errorsDerivative = ComputeErrorDerivatives_(pred, targets, act.actIn[act.actIn.size() - 1], 1.0f, 1.0f);

	EvalNet::Gradients grad;

	m_ann.InitializeGradients(grad);

	m_ann.BackwardPropagateComputeGrad(errorsDerivative, act, grad);

	m_ann.ApplyWeightUpdates(grad, 1.0f, 0.0f);

	InvalidateCache();

	return ((pred - targets).array() * (pred - targets).array()).sum() / targets.rows();
}

float ANNEvaluator::Train(const NNMatrixRM &positions, const NNMatrixRM &targets)
{
	// in this version (where we don't have predictions already) we can simply call ANN's TrainGDM
	float e = m_ann.TrainGDM(positions, targets, 1.0f, 1.0f);

	InvalidateCache();

	return e;
}

void ANNEvaluator::EvaluateForWhiteMatrix(const NNMatrixRM &x, NNMatrixRM &pred, EvalNet::Activations &act)
{
	if (act.act.size() == 0)
	{
		m_ann.InitializeActivations(act);
	}

	pred = m_ann.ForwardPropagate(x, act);
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

	float annOut = m_ann.ForwardPropagateSingle(mappedVec);

	Score nnRet = annOut * EvalFullScale;

	HashStore_(b, nnRet);

	return nnRet;
}

void ANNEvaluator::BatchEvaluateForWhiteImpl(std::vector<Board> &positions, std::vector<Score> &results, Score /*lowerBound*/, Score /*upperBound*/)
{
	// some entries may already be in cache
	// these are the ones we need to evaluate
	std::vector<size_t> toEvaluate;

	results.resize(positions.size());

	for (size_t i = 0; i < positions.size(); ++i)
	{
		auto hashResult = HashProbe_(positions[i]);

		if (hashResult)
		{
			results[i] = *hashResult;
		}
		else
		{
			toEvaluate.push_back(i);
		}
	}

	if (m_convTmp.size() == 0)
	{
		Board b;
		FeaturesConv::ConvertBoardToNN(b, m_convTmp);
	}

	NNMatrixRM xNN(static_cast<int64_t>(toEvaluate.size()), static_cast<int64_t>(m_convTmp.size()));

	for (size_t idx = 0; idx < toEvaluate.size(); ++idx)
	{
		FeaturesConv::ConvertBoardToNN(positions[toEvaluate[idx]], m_convTmp);

		for (size_t i = 0; i < m_convTmp.size(); ++i)
		{
			xNN(idx, i) = m_convTmp[i];
		}
	}

	auto annResults = m_ann.ForwardPropagateFast(xNN);

	for (size_t idx = 0; idx < toEvaluate.size(); ++idx)
	{
		Score result = annResults(idx, 0) * EvalFullScale;

		results[toEvaluate[idx]] = result;

		HashStore_(positions[toEvaluate[idx]], result);
	}
}

void ANNEvaluator::PrintDiag(Board &board)
{
	FeaturesConv::ConvertBoardToNN(board, m_convTmp);

	Eigen::Map<NNVector> mappedVec(&m_convTmp[0], 1, m_convTmp.size());

	std::cout << "Val: " << m_ann.ForwardPropagateSingle(mappedVec) << std::endl;
}

void ANNEvaluator::InvalidateCache()
{
	for (auto &entry : m_evalHash)
	{
		entry.hash = 0;
	}
}

NNMatrixRM ANNEvaluator::BoardsToFeatureRepresentation_(const std::vector<std::string> &positions, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions)
{
	NNMatrixRM ret(positions.size(), featureDescriptions.size());

	{
		ScopedThreadLimiter tlim(8);

		#pragma omp parallel
		{
			std::vector<float> features; // each thread reuses a vector to avoid needless allocation/deallocation

			#pragma omp for
			for (size_t i = 0; i < positions.size(); ++i)
			{
				Board b(positions[i]);
				FeaturesConv::ConvertBoardToNN(b, features);

				if (features.size() != featureDescriptions.size())
				{
					std::stringstream msg;

					msg << "Wrong feature vector size! " << features.size() << " (Expecting: " << featureDescriptions.size() << ")";

					throw std::runtime_error(msg.str());
				}

				ret.row(i) = Eigen::Map<NNMatrixRM>(&features[0], 1, static_cast<int64_t>(features.size()));
			}
		}
	}

	return ret;
}

NNMatrixRM ANNEvaluator::ComputeErrorDerivatives_(
	const NNMatrixRM &predictions,
	const NNMatrixRM &targets,
	const NNMatrixRM &finalLayerActivations,
	float positiveWeight,
	float negativeWeight)
{
	// (targets - predictions) * (-1) * dtanh(act)/dz
	int64_t numExamples = predictions.rows();

	NNMatrixRM ret(numExamples, 1);

	// this takes care of everything except the dtanh(act)/dz term, which we can't really vectorize
	ret = (targets - predictions) * -1.0f;

	// derivative of tanh is 1-tanh^2(x)
	for (int64_t i = 0; i < numExamples; ++i)
	{
		float tanhx = tanh(finalLayerActivations(i, 0));
		ret(i, 0) *= 1.0f - tanhx * tanhx;

		if (ret(i, 0) > 0.0f)
		{
			ret(i, 0) *= positiveWeight;
		}
		else
		{
			ret(i, 0) *= negativeWeight;
		}
	}

	return ret;
}

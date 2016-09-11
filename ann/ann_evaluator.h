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

#ifndef ANN_EVALUATOR_H
#define ANN_EVALUATOR_H

#include <vector>
#include <string>

#include <cmath>

#include "evaluator.h"
#include "ann/ann.h"
#include "ann/features_conv.h"
#include "matrix_ops.h"
#include "consts.h"

//#define EVAL_HASH_STATS
//#define LAZY_EVAL

class ANNEvaluator : public EvaluatorIface
{
public:
	struct EvalHashEntry
	{
		uint64_t hash;
		Score val;
	};

	const static size_t EvalHashSize = 32*MB / sizeof(EvalHashEntry);

	ANNEvaluator(bool eigenOnly = false);

	ANNEvaluator(const std::string &filename);

	void FromString(const std::string &str)
	{
		m_ann.FromString(str);
		InvalidateCache();
	}

	std::string ToString()
	{
		return m_ann.ToString();
	}

	void BuildANN(int64_t inputDims);

	void Serialize(const std::string &filename);

	void Deserialize(const std::string &filename);

	// Targets should be in STM
	float Train(const NNMatrixRM &x, const NNMatrixRM &t);

	Score EvaluateForSTM(Board &b, Score lowerBound = SCORE_MIN, Score upperBound = SCORE_MAX) override;

	Score EvaluateForWhiteImpl(Board &b, Score lowerBound, Score upperBound) override;

	void PrintDiag(Board &board) override;

	void InvalidateCache();

private:
	NNMatrixRM BoardsToFeatureRepresentation_(const std::vector<std::string> &positions, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions);

	Optional<Score> HashProbe_(const Board &b)
	{
#ifdef EVAL_HASH_STATS
		static int64_t queries = 0;
		static int64_t hits = 0;
#endif

		Optional<Score> ret;

		uint64_t hash = b.GetHash();
		EvalHashEntry *entry = &m_evalHash[hash % EvalHashSize];

		if (entry->hash == hash)
		{
			ret = entry->val;
		}

		return ret;
	}

	void HashStore_(const Board &b, Score score)
	{
		uint64_t hash = b.GetHash();

		EvalHashEntry *entry = &m_evalHash[hash % EvalHashSize];

		entry->hash = hash;
		entry->val = score;
	}

	ANN m_ann;

	std::vector<float> m_convTmp;

	std::vector<EvalHashEntry> m_evalHash;
};

#endif // ANN_EVALUATOR_H

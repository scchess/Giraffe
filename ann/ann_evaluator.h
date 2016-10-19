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

	const static size_t EvalHashSize = 8*MB / sizeof(EvalHashEntry);
	const static size_t BatchSize = 8;
	const static size_t EnsembleSize = 8;

	ANNEvaluator(bool eigenOnly = false);

	ANNEvaluator(const std::string &filename);

	void BuildEnsemble();
	void LoadEnsemble(const std::string &baseFilename);
	void SaveEnsemble(const std::string &baseFilename);

	void FromString(const std::string &str)
	{
		m_ann.FromString(str);

		InvalidateCache();
	}

	std::string ToString()
	{
		return m_ann.ToString();
	}

	void BuildANN();

	void Serialize(const std::string &filename);

	void Deserialize(const std::string &filename);

	// Targets should be in STM
	float Train(const NNMatrixRM &x, const NNMatrixRM &t);

	float TrainWithEnsemble(const NNMatrixRM &x, const NNMatrixRM &t);

	// This is only used in training
	NNMatrixRM *EvaluateMatrixWTM(const NNMatrixRM x);

	void ResetOptimizer() { m_ann.ResetOptimizer(); }

	bool IsANNEval() override { return true; }

	Score EvaluateForWhiteImpl(Board &b, Score lowerBound, Score upperBound) override;

	void PrintDiag(Board &board) override;

	void InvalidateCache();

	/* To take advantage of more efficient computation by doing multiple forwards at the same time, the
	 * caller can evaluate using batches. RunBatch() doesn't return anything, but will ensure that everything
	 * in the batch will be in the cache */
	void NewBatch()
	{
		m_currentBatchSize = 0;
		m_batchInput.resize(BatchSize, static_cast<int64_t>(m_convTmp.size()));
		m_batchHashes.resize(BatchSize);
	}

	bool BatchFull()
	{
		return m_currentBatchSize >= BatchSize;
	}

	void AddToBatch(Board &b)
	{
		assert(m_currentBatchSize < BatchSize);

		FeaturesConv::ConvertBoardToNN(b, m_convTmp);

		// we have to map every time because the vector's buffer could have moved
		Eigen::Map<NNVector> mappedVec(&m_convTmp[0], 1, m_convTmp.size());

		m_batchInput.row(m_currentBatchSize) = mappedVec;

		m_batchHashes[m_currentBatchSize] = b.GetHash();

		++m_currentBatchSize;
	}

	void RunBatch()
	{
		if (m_currentBatchSize == 0)
		{
			return;
		}

		NNMatrixRM *results = m_ann.ForwardMultiple(m_batchInput.topRows(m_currentBatchSize));

		for (size_t i = 0; i < m_currentBatchSize; ++i)
		{
			HashStore_(m_batchHashes[i], (*results)(i, 0));
		}
	}

	~ANNEvaluator() {}

private:
	NNMatrixRM BoardsToFeatureRepresentation_(const std::vector<std::string> &positions, const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions);

	Optional<Score> HashProbe_(const Board &b)
	{
		return HashProbe_(b.GetHash());
	}

	Optional<Score> HashProbe_(uint64_t hash)
	{
//#define EVAL_HASH_STATS
#ifdef EVAL_HASH_STATS
		static int64_t queries = 0;
		static int64_t hits = 0;
#endif

		Optional<Score> ret;
		EvalHashEntry *entry = &m_evalHash[hash % EvalHashSize];

		if (entry->hash == hash)
		{
			ret = entry->val;
		}

#ifdef EVAL_HASH_STATS
		if (entry->hash == hash)
		{
			++hits;
		}

		++queries;

		if (queries % 100000 == 0 && queries != 0)
		{
			std::cout << hits << "/" << queries << " (" << (static_cast<float>(hits) / queries) << ")" << std::endl;
		}
#endif

		return ret;
	}

	void HashStore_(const Board &b, Score score)
	{
		HashStore_(b.GetHash(), score);
	}

	void HashStore_(uint64_t hash, Score score)
	{
		EvalHashEntry *entry = &m_evalHash[hash % EvalHashSize];

		entry->hash = hash;
		entry->val = score;
	}

	ANN m_ann;

	std::vector<ANN> m_ensemble;

	std::vector<float> m_convTmp;

	std::vector<EvalHashEntry> m_evalHash;

	size_t m_currentBatchSize = 0;
	NNMatrixRM m_batchInput;
	std::vector<uint64_t> m_batchHashes;
};

#endif // ANN_EVALUATOR_H

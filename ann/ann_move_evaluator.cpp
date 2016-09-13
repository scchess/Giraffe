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

#include "ann_move_evaluator.h"
#include "ann_move_evaluator.h"

#include "random_device.h"
#include "search.h"
#include "static_move_evaluator.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

namespace
{

void TargetsToYNN(const std::vector<float> &trainingTargets, NNMatrixRM &yNN)
{
	yNN.resize(trainingTargets.size(), 1);

	for (int64_t i = 0; i < static_cast<int64_t>(trainingTargets.size()); ++i)
	{
		yNN(i, 0) = trainingTargets[i];
	}
}

}

ANNMoveEvaluator::ANNMoveEvaluator(ANNEvaluator &annEval)
	: m_annEval(annEval)
{
	m_ann = ANN("make_move_evaluator", FeaturesConv::GetMoveNumFeatures());
}

void ANNMoveEvaluator::Train(const std::vector<std::string> &positions, const std::vector<std::string> &bestMoves)
{
	NNMatrixRM trainingSet;
	std::vector<float> trainingTarget;

	// training set size is approx 35 * positionsPerBatch
	size_t positionsPerBatch = std::min<size_t>(positions.size(), 16);

	const static size_t NumIterations = 100000;
	const static size_t IterationsPerPrint = 100;

	auto rng = gRd.MakeMT();
	auto positionDist = std::uniform_int_distribution<size_t>(0, positions.size() - 1);
	auto positionDrawFunc = std::bind(positionDist, rng);

	for (size_t iter = 0; iter < NumIterations; ++iter)
	{
		if ((iter % IterationsPerPrint) == 0)
		{
			std::cout << iter << "/" << NumIterations << std::endl;
		}

		trainingSet.resize(0, 0);
		trainingTarget.clear();

		for (size_t positionNum = 0; positionNum < positionsPerBatch; ++positionNum)
		{
			size_t idx = positionDrawFunc();
			Board pos = Board(positions[idx]);
			Move bestMove = pos.ParseMove(bestMoves[idx]);

			MoveList ml;
			pos.GenerateAllLegalMoves<Board::ALL>(ml);

			NNMatrixRM trainingSetBatch;
			std::vector<float> trainingTargetBatch;

			FeaturesConv::ConvertMovesInfo convInfo;

			GenerateMoveConvInfo_(pos, ml, convInfo);

			FeaturesConv::ConvertMovesToNN(pos, convInfo, ml, trainingSetBatch);

			for (size_t moveNum = 0; moveNum < ml.GetSize(); ++moveNum)
			{
				if (bestMove == ml[moveNum])
				{
					trainingTargetBatch.push_back(1.0f);
				}
				else
				{
					trainingTargetBatch.push_back(0.0f);
				}
			}

			assert(static_cast<size_t>(trainingSetBatch.rows()) == trainingTargetBatch.size());

			#pragma omp critical(trainingSetInsert)
			{
				int64_t origNumExamples = trainingSet.rows();

				NNMatrixRM orig = trainingSet;

				trainingSet.resize(trainingSet.rows() + trainingSetBatch.rows(), trainingSetBatch.cols());

				if (origNumExamples != 0)
				{
					// we have to copy the original over again because resize invalidates everything
					trainingSet.block(0, 0, origNumExamples, trainingSet.cols()) = orig;
				}

				trainingSet.block(origNumExamples, 0, trainingSetBatch.rows(), trainingSet.cols()) = trainingSetBatch;

				trainingTarget.insert(trainingTarget.end(), trainingTargetBatch.begin(), trainingTargetBatch.end());
			}
		}

		NNMatrixRM yNN;
		TargetsToYNN(trainingTarget, yNN);

		assert(trainingSet.rows() == yNN.rows());

		m_ann.Train(trainingSet, yNN);
	}
}

void ANNMoveEvaluator::Test(const std::vector<std::string> &positions, const std::vector<std::string> &bestMoves)
{
	// where in the list is the best move found
	int64_t orderPosCount[100] = { 0 };

	float averageConfidence = 0.0f;

	size_t totalPositions = 0;

	for (size_t posNum = 0; posNum < positions.size(); ++posNum)
	{
		Board board(positions[posNum]);
		Move bestMove = board.ParseMove(bestMoves[posNum]);

		// don't use positions where the best move is a winning capture
		if (SEE::StaticExchangeEvaluation(board, bestMove) > 0)
		{
			continue;
		}

		SearchInfo si;
		MoveInfoList list;
		MoveList ml;

		board.GenerateAllLegalMoves<Board::ALL>(ml);

		for (size_t i = 0; i < ml.GetSize(); ++i)
		{
			MoveInfo mi;
			mi.move = ml[i];
			list.PushBack(mi);
		}

		si.totalNodeBudget = 1000000000;

		EvaluateMoves(board, si, list, ml);

		NormalizeMoveInfoList(list);

		assert(list.GetSize() == ml.GetSize());

		for (size_t i = 0; i < list.GetSize(); ++i)
		{
			if (bestMove == list[i].move)
			{
				if (i < 100)
				{
					++orderPosCount[i];
				}

				averageConfidence += list[i].nodeAllocation;
			}
		}

		++totalPositions;
	}

	averageConfidence /= totalPositions;

	std::cout << "Ordering position: " << std::endl;

	int64_t cCount = 0;

	for (size_t i = 0; i < 20; ++i)
	{
		cCount += orderPosCount[i];

		std::cout << i << ": " << (static_cast<float>(orderPosCount[i]) / totalPositions * 100.0f) << "%" <<
			" (" << (static_cast<float>(cCount) / totalPositions * 100.0f) << ")" << std::endl;
	}

	std::cout << "Average Confidence: " << averageConfidence << std::endl;
}

void ANNMoveEvaluator::EvaluateMoves(Board &board, SearchInfo &si, MoveInfoList &list, MoveList &ml)
{
	if (si.isQS || si.totalNodeBudget < MinimumNodeBudget)
	{
		// delegate to the static evaluator if it's QS, or if we are close to leaf
		// since we don't want to spend more time deciding what to search than actually searching them
		gStaticMoveEvaluator.EvaluateMoves(board, si, list, ml);
		return;
	}

	if (ml.GetSize() == 0)
	{
		return;
	}

	FeaturesConv::ConvertMovesInfo convInfo;

	// we need this even if it's a cache hit, because this is where we compute SEE scores
	GenerateMoveConvInfo_(board, ml, convInfo);

	// we can only cache NN prop results because killers, etc, can change
	using NNCacheEntry = std::pair<uint64_t, NNMatrixRM>;

	static const size_t MevalCacheSize = 65536;
	static std::vector<NNCacheEntry> cache(MevalCacheSize);

	static NNCacheEntry &entry = cache[board.GetHash() % MevalCacheSize];

	if (entry.first != board.GetHash())
	{
		NNMatrixRM xNN;

		FeaturesConv::ConvertMovesToNN(board, convInfo, ml, xNN);

		entry.first = board.GetHash();
		entry.second = *m_ann.ForwardMultiple(xNN);

		// scale to max 1 (NOT normalize)
		entry.second /= entry.second.maxCoeff();
	}

	NNMatrixRM &results = entry.second;

	Score maxSee = std::numeric_limits<Score>::min();

	for (size_t i = 0; i < list.GetSize(); ++i)
	{
		list[i].seeScore = convInfo.see[i];
		list[i].nmSeeScore = convInfo.nmSee[i];

		maxSee = std::max<Score>(maxSee, convInfo.see[i]);
	}

	KillerMoveList killerMoves;

	if (si.killer)
	{
		si.killer->GetKillers(killerMoves, si.ply);
	}

	// whether we should reallocate each move (not interesting moves)
	// using uint8_t to avoid bool specialization
	std::vector<uint8_t> notInteresting(list.GetSize());

	for (size_t i = 0; i < list.GetSize(); ++i)
	{
		notInteresting[i] = false;

		Move mv = list[i].move;

		PieceType promoType = GetPromoType(mv);

		bool isPromo = IsPromotion(mv);
		bool isQueenPromo = (promoType == WQ || promoType == BQ);
		bool isUnderPromo = (isPromo && !isQueenPromo);

		bool isViolent = board.IsViolent(mv);

		if (mv == si.hashMove)
		{
			list[i].nodeAllocation = 3.0f;
		}
		else if (isQueenPromo && list[i].seeScore >= 0)
		{
			list[i].nodeAllocation = 2.0001f;
		}
		else if (isViolent && list[i].seeScore >= 0 && !isUnderPromo)
		{
			list[i].nodeAllocation = 2.0f;
		}
		else if (list[i].seeScore >= 0 && !isUnderPromo)
		{
			notInteresting[i] = true;

			list[i].nodeAllocation = 1.0f; // this will be overwritten later
		}
		else
		{
			notInteresting[i] = true;

			list[i].nodeAllocation = 0.01f; // this will be overwritten as well
		}
	}

	// now we have to figure out the maximum allocation for non-interesting nodes (after normalization)
	float maxNonInterestingNNWeight = 0.0f;
	for (size_t i = 0; i < list.GetSize(); ++i)
	{
		if (notInteresting[i])
		{
			maxNonInterestingNNWeight = std::max<float>(maxNonInterestingNNWeight, results(i, 0));
		}
	}

	float nonInterestingScale = 1.0f / maxNonInterestingNNWeight;

	// killer multipliers based on slots
	const float KillerMultipliers[6] = { 3.0f, 1.5f, 1.2f, 1.2f, 1.2f, 1.2f };

	for (size_t i = 0; i < list.GetSize(); ++i)
	{
		if (notInteresting[i])
		{
			list[i].nodeAllocation = results(i, 0) * nonInterestingScale;

			if (killerMoves.Exists(list[i].move))
			{
				for (size_t slot = 0; slot < killerMoves.GetSize(); ++slot)
				{
					if (killerMoves[slot] == list[i].move)
					{
						// for killer moves, score is based on which slot we are in (lower = better)
						list[i].nodeAllocation *= KillerMultipliers[slot];

						break;
					}
				}
			}

			list[i].nodeAllocation = std::min(list[i].nodeAllocation, 1.0f);
		}
	}

	std::stable_sort(list.begin(), list.end(), [&si, &board, &killerMoves](const MoveInfo &a, const MoveInfo &b)
		{
			if (a.nodeAllocation != b.nodeAllocation)
			{
				return a.nodeAllocation > b.nodeAllocation;
			}
			else
			{
				// sort based on SEE (or another source of score)
				return a.seeScore > b.seeScore;
			}

			//return a.nodeAllocation > b.nodeAllocation;
		}
	);

	NormalizeMoveInfoList(list);
}

void ANNMoveEvaluator::PrintDiag(Board &b)
{
	SearchInfo si;
	si.isQS = false;

	si.totalNodeBudget = 100000;

	MoveInfoList list;

	GenerateAndEvaluateMoves(b, si, list);

	for (auto &mi : list)
	{
		std::cout << b.MoveToAlg(mi.move) << ": " << mi.nodeAllocation << std::endl;
	}
}

void ANNMoveEvaluator::Serialize(const std::string &filename)
{
	m_ann.Save(filename);
}

void ANNMoveEvaluator::Deserialize(const std::string &filename)
{
	m_ann.Load(filename);
}

void ANNMoveEvaluator::GenerateMoveConvInfo_(Board &board, MoveList &ml, FeaturesConv::ConvertMovesInfo &convInfo)
{
	convInfo.see.resize(ml.GetSize());
	convInfo.nmSee.resize(ml.GetSize());

	for (size_t i = 0; i < ml.GetSize(); ++i)
	{
		convInfo.see[i] = SEE::StaticExchangeEvaluation(board, ml[i]);

		convInfo.nmSee[i] = SEE::NMStaticExchangeEvaluation(board, ml[i]);
	}
}

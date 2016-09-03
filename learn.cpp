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

#include "learn.h"

#include <stdexcept>
#include <vector>
#include <sstream>
#include <algorithm>
#include <random>
#include <functional>

#include <cmath>

#include <omp.h>

#include "matrix_ops.h"
#include "board.h"
#include "ann/features_conv.h"
#include "ann/learn_ann.h"
#include "omp_scoped_thread_limiter.h"
#include "eval/eval.h"
#include "history.h"
#include "search.h"
#include "ttable.h"
#include "killer.h"
#include "countermove.h"
#include "random_device.h"
#include "ann/ann_evaluator.h"
#include "move_evaluator.h"
#include "static_move_evaluator.h"
#include "util.h"
#include "stats.h"

namespace
{

using namespace Learn;

std::string getFilename(int64_t iter)
{
	std::stringstream filenameSs;

	filenameSs << "trainingResults/eval" << iter << ".net";

	return filenameSs.str();
}

bool fileExists(const std::string &filename)
{
	std::ifstream is(filename);

	return is.good();
}

}

namespace Learn
{

void TDL(const std::string &positionsFilename)
{
	std::cout << "Starting TDL training..." << std::endl;

	std::ifstream positionsFile(positionsFilename);

	if (!positionsFile)
	{
		throw std::runtime_error(std::string("Cannot open ") + positionsFilename + " for reading");
	}

	// these are the root positions for training (they don't change)
	std::vector<std::string> rootPositions;

	std::string fen;

	std::cout << "Reading FENs..." << std::endl;

	while (std::getline(positionsFile, fen))
	{
		rootPositions.push_back(fen);
		assert(fen != "");
	}

	std::cout << "Positions read: " << rootPositions.size() << std::endl;

	int64_t numFeatures = FeaturesConv::GetNumFeatures();

	std::cout << "Number of features: " << numFeatures << std::endl;

	ANNEvaluator annEval;
	annEval.BuildANN(numFeatures);

	std::cout << "Eval net built" << std::endl;

	int32_t iteration = 0;
	double timeOffset = 0.0f;

	if (fileExists(TrainingLogFileName))
	{
		std::ifstream trainingLogFile(TrainingLogFileName);
		std::string lastWrittenFileName;

		std::string line;
		while (std::getline(trainingLogFile, line))
		{
			if (line.size() > 0)
			{
				std::stringstream ss(line);
				ss >> iteration;
				ss >> lastWrittenFileName;
				ss >> timeOffset;
			}
			else
			{
				break;
			}
		}

		++iteration;
		annEval = ANNEvaluator(lastWrittenFileName);

		std::cout << "Restarting from iteration " << iteration << " last eval file: " << lastWrittenFileName << std::endl;
	}

	double startTime = CurrentTime() - timeOffset;

	std::ofstream trainingLogFile(TrainingLogFileName, iteration == 0 ? std::ofstream::trunc : std::ofstream::app);

	for (; iteration < NumIterations; ++iteration)
	{
		std::cout << "Iteration " << iteration << " ====================================" << std::endl;

		double iterationStartTime = CurrentTime();

		if (iteration == 0)
		{
			auto rng = gRd.MakeMT();
			auto positionDist = std::uniform_int_distribution<size_t>(0, rootPositions.size() - 1);
			auto positionDrawFunc = std::bind(positionDist, rng);

			std::cout << "Bootstrapping using material eval" << std::endl;

			// first iteration is the bootstrap iteration where we don't do any TD, and simply use
			// material eval to bootstrap

			NNMatrixRM trainingBatch(PositionsFirstIteration, numFeatures);
			NNMatrixRM trainingTargets;

			trainingTargets.resize(trainingBatch.rows(), 1);

			std::vector<float> features;

			for (int64_t row = 0; row < trainingBatch.rows(); ++row)
			{
				Board b;
				Score val;

				do
				{
					b = rootPositions[positionDrawFunc()];
					val = Eval::gStaticEvaluator.EvaluateForWhite(b, SCORE_MIN, SCORE_MAX);
				} while (val == 0);

				FeaturesConv::ConvertBoardToNN(b, features);

				trainingBatch.block(row, 0, 1, trainingBatch.cols()) = MapStdVector(features);
				trainingTargets(row, 0) = Eval::gStaticEvaluator.UnScale(val);
			}

			for (size_t i = 0; i < 10; ++i)
			{
				EvalNet::Activations act;
				NNMatrixRM pred;

				for (int64_t start = 0; start < (trainingBatch.rows() - SgdBatchSize); start += SgdBatchSize)
				{
					auto xBlock = trainingBatch.block(start, 0, SgdBatchSize, trainingBatch.cols());
					auto targetsBlock = trainingTargets.block(start, 0, SgdBatchSize, 1);

					annEval.EvaluateForWhiteMatrix(xBlock, pred, act);

					float e = annEval.Train(pred, act, targetsBlock);

					UNUSED(e);

					#if 0
					if (start == 0)
					{
						std::cout << e << std::endl;
					}
					#endif
				}
			}
		}
		else
		{
			std::vector<std::string> positions;
			std::vector<float> targets;

			int64_t numPositionsApprox = iteration < NumEarlyIteartions ? PositionsEarlyIterations : PositionsPerIteration;
			int64_t numRootPositions = numPositionsApprox / HalfMovesToMake;

			std::cout << "Generating training positions..." << std::endl;

			#pragma omp parallel
			{
				Killer killer;
				TTable ttable(1*MB); // we want the ttable to fit in L3
				CounterMove counter;
				History history;

				auto rng = gRd.MakeMT();
				auto positionDist = std::uniform_int_distribution<size_t>(0, rootPositions.size() - 1);
				auto positionDrawFunc = std::bind(positionDist, rng);

				// make a copy of the evaluator because evaluator is not thread-safe (due to caching)
				auto annEvalThread = annEval;

				std::vector<std::string> threadPositions;
				std::vector<float> threadTargets;

				#pragma omp for schedule(dynamic, 256)
				for (int64_t rootPosNum = 0; rootPosNum < numRootPositions; ++rootPosNum)
				{
					Board pos = Board(rootPositions[positionDrawFunc()]);

					ttable.InvalidateAllEntries();

					if (pos.GetGameStatus() == Board::ONGOING && (rootPosNum % 2) == 0)
					{
						// make 1 random move
						// it's very important that we make an odd number of moves, so that if the move is something stupid, the
						// opponent can take advantage of it (and we will learn that this position is bad) before we have a chance to
						// fix it
						MoveList ml;
						pos.GenerateAllLegalMoves<Board::ALL>(ml);

						auto movePickerDist = std::uniform_int_distribution<size_t>(0, ml.GetSize() - 1);

						pos.ApplyMove(ml[movePickerDist(rng)]);
					}

					// if the game was already ended, or has now ended after the random move, skip
					if (pos.GetGameStatus() != Board::ONGOING)
					{
						continue;
					}

					std::vector<std::pair<std::string, float>> playout;

					// make a few moves, and store the leaves of each move into trainingBatch
					for (int64_t moveNum = 0; moveNum < HalfMovesToMake; ++moveNum)
					{
						Search::SearchResult result = Search::SyncSearchNodeLimited(pos, SearchNodeBudget, &annEvalThread, &gStaticMoveEvaluator, &killer, &ttable, &counter, &history);

						Board leaf = pos;
						leaf.ApplyVariation(result.pv);

						Score rootScoreWhite = result.score * (pos.GetSideToMove() == WHITE ? 1 : -1);

						playout.push_back(std::make_pair(leaf.GetFen(), annEvalThread.UnScale(rootScoreWhite)));

						pos.ApplyMove(result.pv[0]);
						killer.MoveMade();
						ttable.AgeTable();
						history.NotifyMoveMade();

						if (pos.GetGameStatus() != Board::ONGOING)
						{
							break;
						}
					}

					for (size_t i = 0; i < playout.size(); ++i)
					{
						float target = playout[i].second;
						float diffWeight = TDLambda;

						for (size_t j = i + 1; j < playout.size(); ++j)
						{
							float scoreDiff = playout[j].second - playout[j - 1].second;
							target += scoreDiff * diffWeight;
							diffWeight *= TDLambda;
						}

						threadPositions.push_back(playout[i].first);
						threadTargets.push_back(target);
					}
				}

				#pragma omp critical (mergeTrainingExamples)
				{
					positions.insert(positions.end(), std::make_move_iterator(threadPositions.begin()), std::make_move_iterator(threadPositions.end()));
					targets.insert(targets.end(), std::make_move_iterator(threadTargets.begin()), std::make_move_iterator(threadTargets.end()));
				}
			}

			std::cout << "Optimizing..." << std::endl;

			double optimizationStartTime = CurrentTime();

			NNMatrixRM trainingBatch(SgdBatchSize, numFeatures);
			NNMatrixRM pred(SgdBatchSize, 1);
			NNMatrixRM targetsBatch(SgdBatchSize, 1);

			EvalNet::Activations act;

			auto trainingPositionRng = gRd.MakeMT();
			auto trainingPositionDist = std::uniform_int_distribution<size_t>(0, positions.size() - 1);
			auto trainingPositionDrawFunc = std::bind(trainingPositionDist, trainingPositionRng);

			float totalError = 0.0f;
			int64_t sgdIterations = positions.size() / SgdBatchSize * SgdEpochs;

			for (int64_t sgdStep = 0; sgdStep < sgdIterations; ++sgdStep)
			{
				#pragma omp parallel
				{
					std::vector<float> featureConvTemp;

					#pragma omp parallel for
					for (int64_t sampleNumInBatch = 0; sampleNumInBatch < SgdBatchSize; ++sampleNumInBatch)
					{
						size_t positionIdx = trainingPositionDrawFunc();
						Board b(positions[positionIdx]);
						FeaturesConv::ConvertBoardToNN(b, featureConvTemp);
						trainingBatch.block(sampleNumInBatch, 0, 1, numFeatures) = MapStdVector(featureConvTemp);
						targetsBatch(sampleNumInBatch, 0) = targets[positionIdx];
					}
				}

				annEval.EvaluateForWhiteMatrix(trainingBatch, pred, act);
				totalError += annEval.Train(pred, act, targetsBatch);
			}

			std::cout << "Average error: " << (totalError / sgdIterations) << std::endl;
			std::cout << "Optimization time: " << (CurrentTime() - optimizationStartTime) << std::endl;
			std::cout << "Total time: " << (CurrentTime() - startTime) << std::endl;
			std::cout << "Iteration took: " << (CurrentTime() - iterationStartTime) << std::endl;
		}

		if ((iteration % EvaluatorSerializeInterval) == 0)
		{
			std::cout << "Serializing " << getFilename(iteration) << "..." << std::endl;

			std::ofstream annOut(getFilename(iteration));

			annEval.Serialize(annOut);

			trainingLogFile << iteration << ' ' << getFilename(iteration) << ' ' << (CurrentTime() - startTime) << std::endl;
			trainingLogFile.flush();
		}
	}
}

}

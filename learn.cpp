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
#include <chrono>
#include "thread_wrapper.h"

#include <cmath>

#include <omp.h>

#include "matrix_ops.h"
#include "board.h"
#include "ann/features_conv.h"
#include "omp_scoped_thread_limiter.h"
#include "eval/eval.h"
#include "history.h"
#include "search.h"
#include "ttable.h"
#include "killer.h"
#include "random_device.h"
#include "ann/ann_evaluator.h"
#include "move_evaluator.h"
#include "static_move_evaluator.h"
#include "gtb.h"
#include "util.h"
#include "stats.h"

namespace
{

using namespace Learn;

std::string getFilename(int64_t iter)
{
	std::stringstream filenameSs;

	filenameSs << "trainingResults/eval" << iter << ".t7";

	return filenameSs.str();
}

bool fileExists(const std::string &filename)
{
	std::ifstream is(filename);

	return is.good();
}

void clearLineAndReturn()
{
	std::cout << "                                     \r";
}

}

namespace Learn
{

void TDL(const std::string &positionsFilename, const std::string &stsFilename)
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
	annEval.BuildANN();
	//annEval.BuildEnsemble();

	std::cout << "Eval net built" << std::endl;

	std::cout << "Loading STS" << std::endl;
	STS sts(stsFilename);
	std::cout << "STS loaded" << std::endl;

	int32_t iteration = 0;
	double timeOffset = 0.0f;

	if (fileExists(TrainingLogFileName))
	{
		std::ifstream trainingLogFile(TrainingLogFileName);
		std::string lastWrittenFileName;

		std::string line;
		while (std::getline(trainingLogFile, line))
		{
			if (line.size() > 1)
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

		if (lastWrittenFileName != "")
		{
			++iteration;

			ANNEvaluator lastEval(lastWrittenFileName);

			annEval.FromString(lastEval.ToString());
			//annEval.LoadEnsemble(lastWrittenFileName);

			std::cout << "Restarting from iteration " << iteration << " last eval file: " << lastWrittenFileName << std::endl;
		}
	}

	std::vector<std::unique_ptr<ANNEvaluator>> threadEvaluators(omp_get_max_threads());

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
			NNMatrixRM trainingTargets(trainingBatch.rows(), 1);

			std::vector<float> features;

			for (int64_t row = 0; row < trainingBatch.rows(); ++row)
			{
				Board b;
				Score val;

				do
				{
					b = rootPositions[positionDrawFunc()];
					val = Eval::gStaticEvaluator.EvaluateForSTM(b, SCORE_MIN, SCORE_MAX);
				} while (val == 0);

				FeaturesConv::ConvertBoardToNN(b, features);

				trainingBatch.block(row, 0, 1, trainingBatch.cols()) = MapStdVector(features);

				if (b.GetSideToMove() == BLACK)
				{
					val *= -1;
				}

				trainingTargets(row, 0) = Eval::gStaticEvaluator.UnScale(val);
			}

			for (size_t i = 0; i < 3; ++i)
			{
				float lossSum = 0.0f;
				int64_t numBatches = 0;
				for (int64_t start = 0; start < (trainingBatch.rows() - SgdBatchSize); start += SgdBatchSize)
				{
					auto xBlock = trainingBatch.block(start, 0, SgdBatchSize, trainingBatch.cols());
					auto targetsBlock = trainingTargets.block(start, 0, SgdBatchSize, 1);
					//lossSum += annEval.TrainWithEnsemble(xBlock, targetsBlock);
					lossSum += annEval.Train(xBlock, targetsBlock);
					++numBatches;
				}
				std::cout << "Epoch " << i << " loss: " << lossSum / numBatches << std::endl;
			}
		}
		else
		{
			//int64_t numPositionsApprox = iteration < NumEarlyIteartions ? PositionsEarlyIterations : PositionsPerIteration;
			int64_t numRootPositions = PositionsPerIteration / HalfMovesToMake;

			auto annParams = annEval.ToString();

			std::vector<std::string> positions;
			std::vector<float> targets;

			#pragma omp parallel
			{
				Killer killer;
				TTable ttable(1*MB); // we want the ttable to fit in L3
				History history;

				ttable.InvalidateAllEntries();

				auto rng = gRd.MakeMT();
				auto positionDist = std::uniform_int_distribution<size_t>(0, rootPositions.size() - 1);
				auto positionDrawFunc = std::bind(positionDist, rng);

				auto thread_id = omp_get_thread_num();
				assert(static_cast<size_t>(thread_id) < threadEvaluators.size());

				if (threadEvaluators[thread_id].get() == nullptr)
				{
					threadEvaluators[thread_id] = std::unique_ptr<ANNEvaluator>(new ANNEvaluator(true /* Eigen-only */));
				}

				// make a copy of the evaluator because evaluator is not thread-safe
				ANNEvaluator &annEvalThread = *threadEvaluators[thread_id];
				annEvalThread.FromString(annParams);

				std::vector<std::string> threadPositions;
				std::vector<float> threadTargets;

				// FEN, search score (from white), leaf color
				std::vector<std::tuple<std::string, float, Color>> playout;

				#pragma omp for schedule(dynamic, 1)
				for (int64_t rootPosNum = 0; rootPosNum < numRootPositions; ++rootPosNum)
				{
					Board pos = Board(rootPositions[positionDrawFunc()]);

					killer.Clear();
					ttable.ClearTable();
					history.Clear();

					if (pos.GetGameStatus() == Board::ONGOING)
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

					playout.clear();

					// make a few moves, and store the leaves of each move into trainingBatch
					for (int64_t moveNum = 0; moveNum < HalfMovesToMake; ++moveNum)
					{
						Search::SearchResult result = Search::SyncSearchNodeLimited(pos, SearchNodeBudget, &annEvalThread, &gStaticMoveEvaluator, &killer, &ttable, &history);

						Board leaf = pos;
						leaf.ApplyVariation(result.pv);

						auto whiteScore = result.score * (pos.GetSideToMove() == WHITE ? 1 : -1);

						playout.push_back(std::make_tuple(leaf.GetFen(), annEvalThread.UnScale(whiteScore), leaf.GetSideToMove()));

						pos.ApplyMove(result.pv[0]);
						killer.MoveMade();
						ttable.AgeTable();
						history.NotifyMoveMade();

						if (pos.GetGameStatus() != Board::ONGOING || IsMateScore(result.score) || IsDrawScore(result.score))
						{
							break;
						}
					}

					for (size_t i = 0; i < (playout.size() - 1); ++i)
					{
						float target = std::get<1>(playout[i]);
						float diffWeight = TDLambda;

						for (size_t j = /*i + 1*/ i + 2; j < playout.size(); /*++j*/ j += 2)
						{
							float scoreDiff = std::get<1>(playout[j]) - std::get<1>(playout[/*j - 1*/ j - 2]);
							target += scoreDiff * diffWeight;
							diffWeight *= TDLambda;
						}

						threadPositions.push_back(std::get<0>(playout[i]));
						threadTargets.push_back(target);

						// push the mirrored position as well
						// Board b(std::get<0>(playout[i]));
						// threadPositions.push_back(b.GetMirroredPosition().GetFen());
						// threadTargets.push_back(-target);
					}
				}

				#pragma omp critical (mergeTrainingExamples)
				{
					positions.insert(positions.end(),
									 std::make_move_iterator(threadPositions.begin()),
									 std::make_move_iterator(threadPositions.end()));
					targets.insert(targets.end(),
								   std::make_move_iterator(threadTargets.begin()),
								   std::make_move_iterator(threadTargets.end()));
				}
			}
			
			double optimizationStartTime = CurrentTime();
			float error = 0.0f;

			NNMatrixRM trainingFeatures(static_cast<int64_t>(positions.size()), numFeatures);

			#pragma omp parallel
			{
				std::vector<float> featureConvTemp;

				#pragma omp for
				for (size_t i = 0; i < positions.size(); ++i)
				{
					Board b(positions[i]);
					FeaturesConv::ConvertBoardToNN(b, featureConvTemp);
					trainingFeatures.block(static_cast<int64_t>(i), 0, 1, numFeatures) = MapStdVector(featureConvTemp);
				}
			}

			annEval.ResetOptimizer();

			int64_t numSgdIterationsPerEpoch = positions.size() / SgdBatchSize + 1;

			NNMatrixRM trainingFeaturesBatch(SgdBatchSize, numFeatures);
			NNMatrixRM targetsBatch(SgdBatchSize, 1);

			auto batchRng = gRd.MakeMT();
			auto batchDist = std::uniform_int_distribution<size_t>(0, positions.size() - 1);
			auto batchDrawFunc = std::bind(batchDist, batchRng);

			for (int64_t epoch = 0; epoch < SgdEpochs; ++epoch)
			{
				float totalError = 0.0f;

				for (int64_t sgdIteration = 0; sgdIteration < numSgdIterationsPerEpoch; ++sgdIteration)
				{
					for (int64_t sample = 0; sample < SgdBatchSize; ++sample)
					{
						int64_t row = batchDrawFunc();
						trainingFeaturesBatch.row(sample) = trainingFeatures.row(row);
						targetsBatch(sample, 0) = targets[row];
					}
				
					error = annEval.Train(trainingFeaturesBatch, targetsBatch);
					totalError += error;
					//error = annEval.TrainWithEnsemble(trainingFeaturesBatch, targetsBatch);
				}

				std::cout << "Epoch error: " << (totalError / numSgdIterationsPerEpoch) << std::endl;
			}


			if ((iteration % EvaluatorSerializeInterval) == 0)
			{
				std::cout << "Optimization time: " << (CurrentTime() - optimizationStartTime) << std::endl;
				std::cout << "Total time: " << (CurrentTime() - startTime) << std::endl;
				std::cout << "Iteration took: " << (CurrentTime() - iterationStartTime) << std::endl;

				std::cout << "Serializing " << getFilename(iteration) << "..." << std::endl;

				annEval.Serialize(getFilename(iteration));
				annEval.SaveEnsemble(getFilename(iteration));

				std::cout << "Testing on STS..." << std::endl;
				auto stsScore = sts.Run(0.1f, &annEval);
				std::cout << "Score: " << stsScore << std::endl;

				trainingLogFile << iteration << ' ' << getFilename(iteration) << ' ' << (CurrentTime() - startTime) << ' ' << stsScore << std::endl;
				trainingLogFile.flush();

				// Sleep a bit to give plot.py time to run. This is purely cosmetic.
				std::this_thread::sleep_for(std::chrono::seconds(2));
			}
		}
	}
}

namespace
{

std::vector<std::string> split(const std::string &s, char delim)
{
	std::vector<std::string> ret;
	std::stringstream ss(s);
	std::string str;

	while (std::getline(ss, str, delim))
	{
		ret.push_back(str);
	}

	return ret;
}

std::string trim(const std::string &input)
{
	auto start = input.find_first_not_of(" \t");
	auto len = input.find_last_not_of(" \t") - start + 1;
	return input.substr(start, len);
}

}

STS::STS(const std::string &filename)
{
	std::ifstream stsFile(filename);

	if (!stsFile)
	{
		std::cerr << "Failed to open " << filename << " for reading" << std::endl;
		assert(false);
	}

	std::string line;
	while (std::getline(stsFile, line))
	{
		STSEntry entry;
		std::stringstream ss(line);
		std::string field;
		while (std::getline(ss, field, ';'))
		{
			field = trim(field);

			if (field.find('\"') == std::string::npos)
			{
				entry.position = Board(field);
			}
			else if (field.substr(0, 2) == "id")
			{
				field = field.substr(4, field.size() - 5);
				entry.id = field;
			}
			else if (field.substr(0, 2) == "c0")
			{
				field = field.substr(4, field.size() - 5);
				auto moveScores = split(field, ',');
				for (auto &s : moveScores)
				{
					auto p = split(s, '=');
					Move mv = entry.position.ParseMove(trim(p[0]));
					entry.moveScores[mv] = ParseStr<int>(trim(p[1]));
				}
			}
		}

		m_entries.push_back(entry);
	}
}

int64_t STS::Run(float maxTime, EvaluatorIface *evaluator)
{
	std::atomic<int> finalScore = {0};

	if (!evaluator->IsANNEval())
	{
		std::cerr << "Only ANN evaluator is supported" << std::endl;
	}

	ANNEvaluator *annEval = reinterpret_cast<ANNEvaluator*>(evaluator);

	std::string evaluatorString = annEval->ToString();

	#pragma omp parallel
	{
		std::unique_ptr<TTable> ttable_u(new TTable(1 * MB));
		Killer killer;
		History history;
		ANNEvaluator threadEval(true);
		threadEval.FromString(evaluatorString);

		#pragma omp barrier

		#pragma omp for
		for (size_t i = 0; i < m_entries.size(); ++i)
		{
			Search::RootSearchContext context;

			context.timeAlloc = { maxTime, maxTime };
			context.searchType = Search::SearchType_makeMove;
			context.nodeBudget = 0;
			context.transpositionTable = ttable_u.get();
			context.killer = &killer;
			context.history = &history;
			context.evaluator = &threadEval;
			context.moveEvaluator = &gStaticMoveEvaluator;
			context.stopRequest = false;

			context.startBoard = m_entries[i].position;

			Search::AsyncSearch search(context);
			search.Start();
			search.Join();

			Move returnedMove = search.GetResult().pv[0];

			if (returnedMove == 0)
			{
				std::cerr << "Search did not return a result!" << std::endl;
			}

			auto it = m_entries[i].moveScores.find(returnedMove);
			if (it != m_entries[i].moveScores.end())
			{
				finalScore += it->second;
			}
		}
	}

	return finalScore;
}

}

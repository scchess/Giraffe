#include "learn.h"

#include <stdexcept>
#include <vector>
#include <sstream>

#include <omp.h>

#include "matrix_ops.h"
#include "board.h"
#include "ann/features_conv.h"
#include "ann/learn_ann.h"
#include "omp_scoped_thread_limiter.h"
#include "eval/eval.h"
#include "search.h"
#include "ttable.h"
#include "killer.h"

namespace
{

std::string getFilename(int64_t iter)
{
	std::stringstream filenameSs;

	filenameSs << "trainingResults/net" << iter << ".dump";

	return filenameSs.str();
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

	while (std::getline(positionsFile, fen) && rootPositions.size() < MaxTrainingPositions)
	{
		rootPositions.push_back(fen);
	}

	std::cout << "Positions read: " << rootPositions.size() << std::endl;

	// these are the leaf positions used in training
	// they are initialized to root positions, but will change in second iteration
	std::vector<std::string> trainingPositions = rootPositions;

	NNMatrixRM trainingTargets(trainingPositions.size(), 1);

	ANNEvaluator annEvaluator;

	std::vector<FeaturesConv::FeatureDescription> featureDescriptions;

	FeaturesConv::ConvertBoardToNN(Board(), featureDescriptions);

	NNMatrixRM boardsInFeatureRepresentation(static_cast<int64_t>(trainingPositions.size()), static_cast<int64_t>(featureDescriptions.size()));

	int64_t iter = 0;

	// try to read existing dumps to continue where we left off
	for (; iter < NumIterations; ++iter)
	{
		std::string filename = getFilename(iter);

		std::ifstream dump(filename);

		if (dump.is_open())
		{
			ANN newAnn = DeserializeNet(dump);
			annEvaluator.GetANN() = newAnn;
			annEvaluator.Calibrate();
		}
		else
		{
			break;
		}
	}

	if (iter != 0)
	{
		std::cout << "Continuing from iteration " << iter << std::endl;
	}

	for (; iter < NumIterations; ++iter)
	{
		std::cout << "======= Iteration: " << iter << " =======" << std::endl;

		// first we generate new labels
		// if this is the first iteration, we use static material labels
		if (iter == 0)
		{
			std::cout << "Labelling using static evaluation..." << std::endl;
			#pragma omp parallel for
			for (size_t i = 0; i < trainingPositions.size(); ++i)
			{
				Board b(trainingPositions[i]);
				Score val = Eval::gStaticEvaluator.EvaluateForWhite(b, SCORE_MIN, SCORE_MAX);
				trainingTargets(i, 0) = val;
			}
		}
		else
		{
			std::cout << "Labelling using TDLeaf..." << std::endl;

			size_t positionsProcessed = 0;
			double timeStart = CurrentTime();
			double timeLastPrint = CurrentTime();

			#pragma omp parallel
			{
				// each thread has her own ttable and killers, to save on page faults and allocations/deallocations
				Killer thread_killer;
				TTable thread_ttable(1*MB); // we want the ttable to fit in L3

				// we are being paranoid here - it's possible tha the memory we get happens to be where a TTable used to be,
				// in which case we will have many valid entries with wrong scores (since evaluator changed)
				thread_ttable.InvalidateAllEntries();

				// each thread makes a copy of the evaluator to reduce sharing
				ANNEvaluator thread_annEvaluator = annEvaluator;

				#pragma omp for
				for (size_t i = 0; i < rootPositions.size(); ++i)
				{
					thread_ttable.ClearTable(); // this is a cheap clear that simply ages the table a bunch so all new positions have higher priority

					Board rootPos(rootPositions[i]);
					Search::SearchResult rootResult = Search::SyncSearchDepthLimited(rootPos, 2, &thread_annEvaluator, &thread_killer, &thread_ttable);

					Board leafPos = rootPos;
					leafPos.ApplyVariation(rootResult.pv);

					float leafScore = thread_annEvaluator.EvaluateForWhite(leafPos); // this should theoretically be the same as the search result, except for mates, etc

					trainingPositions[i] = leafPos.GetFen();

					if (rootResult.pv.size() > 0)
					{
						rootPos.ApplyMove(rootResult.pv[0]);
						thread_killer.MoveMade();
						thread_ttable.AgeTable();

						// now we compute the error by making a few moves
						float accumulatedError = 0.0f;
						float lastScore = leafScore;
						float discount = 1.0f;

						for (int64_t m = 0; m < HalfMovesToMake; ++m)
						{
							Search::SearchResult result = Search::SyncSearchDepthLimited(rootPos, 2, &thread_annEvaluator, &thread_killer, &thread_ttable);

							float scoreWhite = result.score * (rootPos.GetSideToMove() == WHITE ? 1.0f : -1.0f);

							// compute error contribution
							accumulatedError += discount * (scoreWhite - lastScore);
							lastScore = scoreWhite;
							discount *= Lambda;

							if ((rootPos.GetGameStatus() != Board::ONGOING) || (result.pv.size() == 0))
							{
								break;
							}

							rootPos.ApplyMove(result.pv[0]);
							thread_killer.MoveMade();
							thread_ttable.AgeTable();
						}

						accumulatedError = std::max(accumulatedError, -MaxError);
						accumulatedError = std::min(accumulatedError, MaxError);

						trainingTargets(i, 0) = leafScore + accumulatedError;
					}
					else
					{
						// if PV is empty, this is an end position, and we don't need to train it
						trainingTargets(i, 0) = leafScore;
					}

					if (omp_get_thread_num() == 0)
					{
						float timeSinceLastPrint = CurrentTime() - timeLastPrint;
						if (timeSinceLastPrint > 15.0f)
						{
							double timeElapsed = CurrentTime() - timeStart;
							std::cout << "Processed: " << positionsProcessed << " Positions/s: " << (positionsProcessed / timeElapsed) << std::endl;
							timeLastPrint = CurrentTime();
						}
					}

					#pragma omp atomic
					++positionsProcessed;
				}
			}
		}

		std::cout << "Converting boards to features..." << std::endl;

		{
			ScopedThreadLimiter tlim(8);

			#pragma omp parallel
			{
				std::vector<float> features; // each thread reuses a vector to avoid needless allocation/deallocation

				#pragma omp for
				for (size_t i = 0; i < trainingPositions.size(); ++i)
				{
					FeaturesConv::ConvertBoardToNN(Board(trainingPositions[i]), features);

					if (features.size() != featureDescriptions.size())
					{
						std::stringstream msg;

						msg << "Wrong feature vector size! " << features.size() << " (Expecting: " << featureDescriptions.size() << ")";

						throw std::runtime_error(msg.str());
					}

					boardsInFeatureRepresentation.row(i) = Eigen::Map<NNMatrixRM>(&features[0], 1, static_cast<int64_t>(features.size()));
				}
			}
		}

		ANN newAnn = LearnAnn::TrainANN(boardsInFeatureRepresentation, trainingTargets, std::string("x_5M_nodiag.features"));

		std::ofstream annOut(getFilename(iter));

		SerializeNet(newAnn, annOut);

		annEvaluator.GetANN() = newAnn;
		annEvaluator.Calibrate();
	}
}

}

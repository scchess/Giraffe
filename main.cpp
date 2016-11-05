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

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "thread_wrapper.h"
#include "mutex_wrapper.h"

#include <cstdint>

#include "magic_moves.h"
#include "board_consts.h"
#include "move.h"
#include "board.h"
#include "eval/eval.h"
#include "see.h"
#include "search.h"
#include "backend.h"
#include "chessclock.h"
#include "util.h"
#include "ann/eigen_ann.h"
#include "ann/features_conv.h"
#include "ann/ann_evaluator.h"
#include "ann/ann_move_evaluator.h"
#include "learn.h"
#include "zobrist.h"
#include "gtb.h"
#include "move_evaluator.h"
#include "static_move_evaluator.h"
#include "move_stats/move_stats.h"

#include "Eigen/Dense"

const std::string EvalNetFilename = "eval.t7";
const std::string MoveEvalNetFilename = "meval.t7";
const std::string InitFileName = "init.txt";

std::string gVersion;

void GetGiraffeVersion()
{
	std::ifstream verFile("version.txt");

	if (verFile.is_open())
	{
		std::getline(verFile, gVersion);

		std::cout << "# Version: " << gVersion << std::endl;
	}
#ifdef HGVERSION
	else
	{
		gVersion = HGVERSION;
		std::cout << "# Version: " << HGVERSION << std::endl;
	}
#endif
}

void InitializeNetworks(ANNEvaluator &evaluator, ANNMoveEvaluator &mevaluator)
{
	std::ifstream evalNet(EvalNetFilename);

	if (evalNet)
	{
		evaluator.Deserialize(EvalNetFilename);
	}

	std::ifstream mevalNet(MoveEvalNetFilename);

	if (mevalNet)
	{
		mevaluator.Deserialize(MoveEvalNetFilename);
	}

	std::cout << GTB::Init();
}

// fast initialization steps that can be done in main thread
void Initialize()
{
	std::cout << "# Using " << omp_get_max_threads() << " OpenMP thread(s)" << std::endl;

	GetGiraffeVersion();

#ifdef DEBUG
	std::cout << "# Running in debug mode" << std::endl;
#else
	std::cout << "# Running in release mode" << std::endl;
#endif

	Eigen::initParallel();

	// set Eigen to use 1 thread because we are doing OpenMP here
	Eigen::setNbThreads(1);

	// disable nested parallelism since we don't need it, and disabling it
	// makes managing number of threads easier
	omp_set_nested(0);

	// turn off IO buffering
	std::cout.setf(std::ios::unitbuf);

	initmagicmoves();
	BoardConstsInit();
	InitializeZobrist();
}

int main(int argc, char **argv)
{
	Initialize();

	ANNEvaluator evaluator;

	ANNMoveEvaluator mevaluator(evaluator);

	InitializeNetworks(evaluator, mevaluator);

	Backend backend;

	// if eval.t7 exists, use the ANN evaluator
	// if both eval.t7 and meval.t7 exist, use the ANN move evaluator

	if (!FileReadable(EvalNetFilename) && argc == 1)
	{
		// in normal playing mode (giraffe run without argument), we should
		// refuse to run without the net files
		std::cout << "tellusererror " << EvalNetFilename << " not found in the working directory. Aborting." << std::endl;
		return 0;
	}

	if (FileReadable(EvalNetFilename))
	{
		backend.SetEvaluator(&evaluator);

		std::cout << "# Using ANN evaluator" << std::endl;

		if (FileReadable(MoveEvalNetFilename))
		{
			std::cout << "# Using ANN move evaluator" << std::endl;
			backend.SetMoveEvaluator(&mevaluator);
		}
		else
		{
			std::cout << "# Using static move evaluator" << std::endl;
			backend.SetMoveEvaluator(&gStaticMoveEvaluator);
		}
	}
	else
	{
		std::cout << "# Using static evaluator" << std::endl;
		std::cout << "# Using static move evaluator" << std::endl;

		backend.SetEvaluator(&Eval::gStaticEvaluator);
		backend.SetMoveEvaluator(&gStaticMoveEvaluator);
	}

	// first we handle special operation modes
	if (argc >= 2 && std::string(argv[1]) == "tdl")
	{
		if (argc < 4)
		{
			std::cout << "Usage: " << argv[0] << " tdl positions sts_filename" << std::endl;
			return 0;
		}

		Learn::TDL(argv[2], argv[3]);

		return 0;
	}
	else if (argc >= 2 && std::string(argv[1]) == "conv")
	{
		if (argc < 3)
		{
			std::cout << "Usage: " << argv[0] << " conv FEN" << std::endl;
			return 0;
		}

		std::stringstream ss;

		for (int i = 2; i < argc; ++i)
		{
			ss << argv[i] << ' ';
		}

		Board b(ss.str());

		std::vector<FeaturesConv::FeatureDescription> ret;
		FeaturesConv::ConvertBoardToNN(b, ret);

		for (auto &fd : ret)
		{
			std::cout << fd.ToString() << std::endl;
		}

		return 0;
	}
	else if (argc >= 2 && std::string(argv[1]) == "mconv")
	{
		if (argc < 3)
		{
			std::cout << "Usage: " << argv[0] << " mconv FEN" << std::endl;
			return 0;
		}

		std::stringstream ss;

		for (int i = 2; i < argc; ++i)
		{
			ss << argv[i] << ' ';
		}

		Board b(ss.str());

		MoveList moves;
		b.GenerateAllLegalMoves<Board::ALL>(moves);

		NNMatrixRM ret;

		FeaturesConv::ConvertMovesInfo convInfo;

		FeaturesConv::ConvertMovesToNN(b, convInfo, moves, ret);

		for (int64_t row = 0; row < ret.rows(); ++row)
		{
			for (int64_t col = 0; col < ret.cols(); ++col)
			{
				std::cout << ret(row, col) << ' ';
			}
			std::cout << std::endl;
		}

		return 0;
	}
	else if (argc >= 2 && std::string(argv[1]) == "bench")
	{
		double startTime = CurrentTime();

		static const NodeBudget BenchNodeBudget = 64*1024*1024;

		Search::SyncSearchNodeLimited(Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), BenchNodeBudget, backend.GetEvaluator(), backend.GetMoveEvaluator());
		Search::SyncSearchNodeLimited(Board("2r2rk1/pp3pp1/b2Pp3/P1Q4p/RPqN2n1/8/2P2PPP/2B1R1K1 w - - 0 1"), BenchNodeBudget, backend.GetEvaluator(), backend.GetMoveEvaluator());
		Search::SyncSearchNodeLimited(Board("8/1nr3pk/p3p1r1/4p3/P3P1q1/4PR1N/3Q2PK/5R2 w - - 0 1"), BenchNodeBudget, backend.GetEvaluator(), backend.GetMoveEvaluator());
		Search::SyncSearchNodeLimited(Board("5R2/8/7r/7P/5RPK/1k6/4r3/8 w - - 0 1"), BenchNodeBudget, backend.GetEvaluator(), backend.GetMoveEvaluator());
		Search::SyncSearchNodeLimited(Board("r5k1/2p2pp1/1nppr2p/8/p2PPp2/PPP2P1P/3N2P1/R3RK2 w - - 0 1"), BenchNodeBudget, backend.GetEvaluator(), backend.GetMoveEvaluator());
		Search::SyncSearchNodeLimited(Board("8/R7/8/1k6/1p1Bq3/8/4NK2/8 w - - 0 1"), BenchNodeBudget, backend.GetEvaluator(), backend.GetMoveEvaluator());

		std::cout << "Time: " << (CurrentTime() - startTime) << "s" << std::endl;

		return 0;
	}
	else if (argc >= 2 && std::string(argv[1]) == "sample_internal")
	{
		// MUST UNCOMMENT "#define SAMPLING" in static move evaluator

		if (argc < 4)
		{
			std::cout << "Usage: " << argv[0] << " sample_internal <EPD/FEN file> <output file>" << std::endl;
			return 0;
		}

		std::ifstream infile(argv[2]);
		std::ofstream outfile(argv[3]);

		if (!infile)
		{
			std::cerr << "Failed to open " << argv[2] << " for reading" << std::endl;
			return 1;
		}

		std::string fen;
		std::vector<std::string> fens;
		static const uint64_t maxPositions = 5000000;
		uint64_t numPositions = 0;
		while (std::getline(infile, fen) && numPositions < maxPositions)
		{
			fens.push_back(fen);
			++numPositions;
		}

		std::string evaluatorParams = evaluator.ToString();

		#pragma omp parallel
		{
			ANNEvaluator evaluatorCopy;
			evaluatorCopy.FromString(evaluatorParams);

			#pragma omp for
			for (size_t i = 0; i < fens.size(); ++i)
			{
				Board b(fens[i]);

				Search::SyncSearchNodeLimited(b, 1000, &evaluatorCopy, &gStaticMoveEvaluator, nullptr, nullptr);
			}
		}

		for (const auto &pos : gStaticMoveEvaluator.samples)
		{
			outfile << pos << std::endl;
		}

		return 0;
	}
	else if (argc >= 2 && std::string(argv[1]) == "label_bm")
	{
		if (argc < 4)
		{
			std::cout << "Usage: " << argv[0] << " label_bm <EPD/FEN file> <output file>" << std::endl;
			return 0;
		}

		std::ifstream infile(argv[2]);
		std::ofstream outfile(argv[3]);

		if (!infile)
		{
			std::cerr << "Failed to open " << argv[2] << " for reading" << std::endl;
			return 1;
		}

		std::string fen;
		std::vector<std::string> fens;
		static const uint64_t maxPositions = 5000000;
		uint64_t numPositions = 0;
		while (std::getline(infile, fen) && numPositions < maxPositions)
		{
			Board b(fen);

			if (b.GetGameStatus() != Board::ONGOING)
			{
				continue;
			}

			fens.push_back(fen);
			++numPositions;
		}

		std::vector<std::string> bm(fens.size());

		uint64_t numPositionsDone = 0;

		double lastPrintTime = CurrentTime();
		size_t lastDoneCount = 0;

		std::string evaluatorParams = evaluator.ToString();

		#pragma omp parallel
		{
			ANNEvaluator evaluatorCopy;
			evaluatorCopy.FromString(evaluatorParams);

			uint64_t numThreadPositionsDone = 0;

			#pragma omp for schedule(dynamic, 4096)
			for (size_t i = 0; i < fens.size(); ++i)
			{
				Board b(fens[i]);

				Search::SearchResult result = Search::SyncSearchNodeLimited(b, 1000, &evaluatorCopy, &gStaticMoveEvaluator, nullptr, nullptr);

				bm[i] = b.MoveToAlg(result.pv[0]);

				++numThreadPositionsDone;

				if (numThreadPositionsDone % 256 == 0)
				{
					#pragma omp critical(numPositionsAndOutputFileUpdate)
					{
						numPositionsDone += numThreadPositionsDone;

						if (omp_get_thread_num() == 0)
						{
							double currentTime = CurrentTime();
							double timeDiff = currentTime - lastPrintTime;
							if (timeDiff > 1.0)
							{
								std::cout << numPositionsDone << '/' << fens.size() << std::endl;
								std::cout << "Positions per second: " << static_cast<double>(numPositionsDone - lastDoneCount) / timeDiff << std::endl;

								lastPrintTime = currentTime;
								lastDoneCount = numPositionsDone;
							}
						}
					}

					numThreadPositionsDone = 0;
				}
			}
		}

		for (size_t i = 0; i < fens.size(); ++i)
		{
			outfile << fens[i] << std::endl;
			outfile << bm[i] << std::endl;
		}

		return 0;
	}
	else if (argc >= 2 && std::string(argv[1]) == "train_move_eval")
	{
		if (argc < 4)
		{
			std::cout << "Usage: " << argv[0] << " train_move_eval <EPD/FEN file> <output file>" << std::endl;
			return 0;
		}

		std::ifstream infile(argv[2]);

		if (!infile)
		{
			std::cerr << "Failed to open " << argv[2] << " for reading" << std::endl;
			return 1;
		}

		std::cout << "Reading positions from " << argv[2] << std::endl;

		std::string fen;
		std::string bestMove;
		std::vector<std::string> fens;
		std::vector<std::string> bestMoves;
		static const uint64_t MaxPositions = 5000000;
		uint64_t numPositions = 0;
		while (std::getline(infile, fen) && std::getline(infile, bestMove) && numPositions < MaxPositions)
		{
			Board b(fen);

			if (b.GetGameStatus() != Board::ONGOING)
			{
				continue;
			}

			fens.push_back(fen);
			bestMoves.push_back(bestMove);

			++numPositions;
		}

		assert(bestMoves.size() == fens.size());

		// now we split a part of it out into a withheld test set
		size_t numTrainExamples = fens.size() * 0.9f;
		std::vector<std::string> fensTest(fens.begin() + numTrainExamples, fens.end());
		std::vector<std::string> bestMovesTest(bestMoves.begin() + numTrainExamples, bestMoves.end());

		static const uint64_t MaxTestingPositions = 10000;

		if (fensTest.size() > MaxTestingPositions)
		{
			fensTest.resize(MaxTestingPositions);
			bestMovesTest.resize(MaxTestingPositions);
		}

		fens.resize(numTrainExamples);
		bestMoves.resize(numTrainExamples);

		std::cout << "Num training examples: " << numTrainExamples << std::endl;
		std::cout << "Num testing examples: " << fensTest.size() << std::endl;

		std::cout << "Starting training" << std::endl;

		ANNMoveEvaluator meval(evaluator);

		meval.Train(fens, bestMoves);

		meval.Test(fensTest, bestMovesTest);

		meval.Serialize(argv[3]);

		return 0;
	}
	else if (argc >= 2 && std::string(argv[1]) == "move_stats")
	{
		if (argc < 3)
		{
			std::cout << "Usage: " << argv[0] << " move_stats <labeled FEN>" << std::endl;
			return 0;
		}

		MoveStats::ProcessStats(argv[2]);

		return 0;
	}

	std::ifstream initFile(InitFileName);

	while (true)
	{
		std::string lineStr;
		if (std::getline(initFile, lineStr))
		{
			std::cout << "# From init file: " << lineStr << std::endl;
		}
		else
		{
			std::getline(std::cin, lineStr);
		}

		std::stringstream line(lineStr);

		// we set usermove=1, so all commands from xboard start with a unique word
		std::string cmd;
		line >> cmd;

		if (cmd == "xboard") {} // ignore since we only support xboard mode anyways
		else if (cmd == "protover")
		{
			int32_t ver;
			line >> ver;

			if (ver >= 2)
			{
				std::string name = "Giraffe";
				if (gVersion != "")
				{
					name += " ";
					name += gVersion;
				}

				std::cout << "feature ping=1 setboard=1 playother=0 san=0 usermove=1 time=1 draw=0 sigint=0 sigterm=0 "
							 "reuse=1 analyze=1 myname=\"" << name << "\" variants=normal colors=0 ics=0 name=0 pause=0 nps=0 "
							 "debug=1 memory=0 smp=0 done=0" << std::endl;

				std::cout << "feature option=\"GaviotaTbPath -path .\"" << std::endl;

				std::cout << "feature done=1" << std::endl;
			}
		}
		else if (cmd == "accepted") {}
		else if (cmd == "rejected") {}
		else if (cmd == "new")
		{
			backend.NewGame();
			backend.SetMaxDepth(0);
		}
		else if (cmd == "setboard")
		{
			std::string fen;
			std::getline(line, fen);
			backend.SetBoard(fen);
		}
		else if (cmd == "quit")
		{
			break;
		}
		else if (cmd == "random") {}
		else if (cmd == "force")
		{
			backend.Force();
		}
		else if (cmd == "go")
		{
			backend.Go();
		}
		else if (cmd == "level")
		{
			int32_t movesPerPeriod;

			double base;

			// base is a little complicated, because it can either be minutes or minutes:seconds
			// so we read into a string first before figuring out what to do with it
			std::string baseStr;

			double inc;

			line >> movesPerPeriod;
			line >> baseStr;
			line >> inc;

			if (baseStr.find(':') == std::string::npos)
			{
				sscanf(baseStr.c_str(), "%lf", &base);
				base *= 60.0;
			}
			else
			{
				double minutes;
				double seconds;

				sscanf(baseStr.c_str(), "%lf:%lf", &minutes, &seconds);
				base = minutes * 60.0 + seconds;
			}

			ChessClock cc(ChessClock::CONVENTIONAL_INCREMENTAL_MODE, movesPerPeriod, base, inc);

			backend.SetTimeControl(cc);
		}
		else if (cmd == "st")
		{
			double t;
			line >> t;
			backend.SetTimeControl(ChessClock(ChessClock::EXACT_MODE, 0, 0.0, t));
		}
		else if (cmd == "sd")
		{
			int32_t maxDepth;
			line >> maxDepth;
			backend.SetMaxDepth(maxDepth);
		}
		else if (cmd == "time")
		{
			double t;
			line >> t;
			t /= 100.0;

			backend.AdjustEngineTime(t);
		}
		else if (cmd == "otim")
		{
			double t;
			line >> t;
			t /= 100.0;

			backend.AdjustOpponentTime(t);
		}
		else if (cmd == "usermove")
		{
			std::string mv;
			line >> mv;
			backend.Usermove(mv);
		}
		else if (cmd == "?") {}
		else if (cmd == "result")
		{
			backend.NewGame();
		}
		else if (cmd == "ping")
		{
			int32_t num;
			line >> num;
			std::cout << "pong " << num << std::endl;
		}
		else if (cmd == "hint") {}
		else if (cmd == "undo")
		{
			backend.Undo(1);
		}
		else if (cmd == "remove")
		{
			backend.Undo(2);
		}
		else if (cmd == "hard")
		{
			backend.SetPondering(true);
		}
		else if (cmd == "easy")
		{
			backend.SetPondering(false);
		}
		else if (cmd == "post")
		{
			backend.SetShowThinking(true);
		}
		else if (cmd == "nopost")
		{
			backend.SetShowThinking(false);
		}
		else if (cmd == "analyze")
		{
			backend.SetAnalyzing(true);
		}
		else if (cmd == "exit")
		{
			backend.SetAnalyzing(false);
		}
		else if (cmd == "computer") {}
		else if (cmd == "printboard")
		{
			// for debugging, not in xboard protocol
			backend.DebugPrintBoard();

			MoveList moves;
			backend.GetBoard().GenerateAllLegalMoves<Board::ALL>(moves);

			// Print legal moves and test the generator and parser while we are at it
			// for (const auto &mv : moves)
			// {
			// 	std::string str = backend.GetBoard().MoveToAlg(mv, Board::SAN);
			// 	assert(backend.GetBoard().ParseMove(str) == mv);
			// 	std::cout << str << std::endl;
			// }
		}
		else if (cmd == "mirror")
		{
			auto board = backend.GetBoard().GetMirroredPosition();
			std::cout << board.PrintBoard() << std::endl;
		}
		else if (cmd == "perft")
		{
			int32_t depth;
			line >> depth;
			backend.DebugRunPerft(depth);
		}
		else if (cmd == "perft_with_null")
		{
			// special perft that also tries null moves (but doesn't count them)
			// this is for debugging null make/unmake
			int32_t depth;
			line >> depth;
			backend.DebugRunPerftWithNull(depth);
		}
		else if (cmd == "eval")
		{
			backend.PrintDebugEval();
		}
		else if (cmd == "meval")
		{
			backend.PrintDebugMoveEval();
		}
		else if (cmd == "gtb")
		{
			std::cout << backend.DebugGTB() << std::endl;
		}
		else if (cmd == "runtests")
		{
			//SEE::DebugRunSeeTests();
			//DebugRunPerftTests();
			DebugRunSANTests();
			std::cout << "All passed!" << std::endl;
		}
		else if (cmd == "gee")
		{
			std::vector<Move> pv;

			Board b = backend.GetBoard();

			SEE::GlobalExchangeEvaluation(b, pv);

			for (size_t i = 0; i < pv.size(); ++i)
			{
				std::cout << b.MoveToAlg(pv[i]) << ' ';
				b.ApplyMove(pv[i]);
			}

			std::cout << std::endl;
		}
		else if (cmd == "atkmaps")
		{
			Board b = backend.GetBoard();

			PieceType whiteAttackers[64];
			PieceType blackAttackers[64];

			uint8_t whiteNumAttackers[64];
			uint8_t blackNumAttackers[64];

			b.ComputeLeastValuableAttackers(whiteAttackers, whiteNumAttackers, WHITE);
			b.ComputeLeastValuableAttackers(blackAttackers, blackNumAttackers, BLACK);

			auto printAtkBoardFcn = [](PieceType attackers[64])
			{
				for (int y = 7; y >= 0; --y)
				{
					std::cout << "   ---------------------------------" << std::endl;
					std::cout << " " << (y + 1) << " |";

					for (int x = 0; x <= 7; ++x)
					{
							std::cout << " " << PieceTypeToChar(attackers[Sq(x, y)]) << " |";
					}

					std::cout << std::endl;
				}

				std::cout << "   ---------------------------------" << std::endl;
			};

			std::cout << "White:" << std::endl;
			printAtkBoardFcn(whiteAttackers);

			std::cout << "Black:" << std::endl;
			printAtkBoardFcn(blackAttackers);
		}
		else if (cmd == "option")
		{
			std::string lineStr;

			std::getline(line, lineStr);

			if (lineStr.find('=') != std::string::npos)
			{
				std::string optionName = lineStr.substr(0, lineStr.find_first_of('='));
				std::string optionValue = lineStr.substr(lineStr.find_first_of('=') + 1, lineStr.size());

				// remove leading and trailing whitespaces
				optionName = std::regex_replace(optionName, std::regex("^ +"), "");
				optionName = std::regex_replace(optionName, std::regex(" +$"), "");

				optionValue = std::regex_replace(optionValue, std::regex("^ +"), "");
				optionValue = std::regex_replace(optionValue, std::regex(" +$"), "");

				// for value, we additionally want to remove quotes
				optionValue = std::regex_replace(optionValue, std::regex("^\"+"), "");
				optionValue = std::regex_replace(optionValue, std::regex("\"+$"), "");

				if (optionName == "GaviotaTbPath")
				{
					std::cout << GTB::Init(optionValue) << std::endl;
				}
				else
				{
					std::cout << "Error: Unknown option - " << optionName << std::endl;
				}
			}
			else
			{
				std::cout << "Error: option requires value" << std::endl;
			}
		}
		else if (cmd == "runsts")
		{
			std::string filename;
			float timePerPosition;
			line >> filename >> timePerPosition;

			Learn::STS sts(filename);
			auto score = sts.Run(timePerPosition, &evaluator);

			std::cout << "Score: " << score << std::endl;
		}
		else if (backend.IsAMove(cmd))
		{
			backend.Usermove(cmd);
		}
		else
		{
			std::cout << "Error (unknown command): " << cmd << std::endl;
		}
	}

	backend.Quit();
	GTB::DeInit();
}

/*
	Copyright (C) 2016 Matthew Lai

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

#include "move_stats.h"

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace MoveStats
{

void ProcessStats(const std::string &filename)
{
	std::ifstream infile(filename);

	if (!infile)
	{
		std::cerr << "Failed to open " << filename << " for reading." << std::endl;
		return;
	}

	uint64_t positionsProcessed = 0;

	std::string line;

	std::vector<FilterIf*> filters;

	SEEFilter seeFilterPos(SEEFilter::POS); filters.push_back(&seeFilterPos);
	SEEFilter seeFilterNeutral(SEEFilter::NEUTRAL); filters.push_back(&seeFilterNeutral);
	SEEFilter seeFilterNeg(SEEFilter::NEG); filters.push_back(&seeFilterNeg);

	HighestSEEFilter highestSeeFilter; filters.push_back(&highestSeeFilter);

	IsCaptureFilter isCaptureFilter; filters.push_back(&isCaptureFilter);
	FilterNot notCaptureFilter(&isCaptureFilter); filters.push_back(&notCaptureFilter);

	PromotionFilter isPromotionFilter; filters.push_back(&isPromotionFilter);
	PromotionTypeFilter isQPromotionFilter(Q); filters.push_back(&isQPromotionFilter);

	GamePhaseFilter isEndGame(6, GamePhaseFilter::LESS_THAN); filters.push_back(&isEndGame);
	GamePhaseFilter isOpening(10, GamePhaseFilter::MORE_THAN_OR_EQUAL); filters.push_back(&isOpening);

	EscapeFilter isEscape; filters.push_back(&isEscape);

	std::map<PieceType, PieceTypeFilter> pieceTypeFilters;
	pieceTypeFilters.emplace(K, K);
	pieceTypeFilters.emplace(Q, Q);
	pieceTypeFilters.emplace(R, R);
	pieceTypeFilters.emplace(B, B);
	pieceTypeFilters.emplace(N, N);
	pieceTypeFilters.emplace(P, P);

	for (auto &x : pieceTypeFilters)
	{
		filters.push_back(&x.second);
	}

	while (infile && positionsProcessed < 100000)
	{
		std::getline(infile, line);
		Board board(line);
		std::getline(infile, line);
		Move bestMove = board.ParseMove(line);

		for (auto &filter : filters)
		{
			filter->RunFilter(board, bestMove);
		}

		++positionsProcessed;

		if (positionsProcessed % 10000 == 0)
		{
			std::cout << positionsProcessed << " positions processed." << std::endl;
		}
	}

	auto SEEPositiveOrNeutral = seeFilterPos.stats() || seeFilterNeutral.stats();

	std::cout << "+SEE captures: " << std::endl;
	auto posSeeCaptures = isCaptureFilter.stats() && seeFilterPos.stats();
	std::cout << (posSeeCaptures).ToString() << std::endl;

	std::cout << "=SEE captures: " << std::endl;
	std::cout << (isCaptureFilter.stats() && seeFilterNeutral.stats()).ToString() << std::endl;

	std::cout << "=SEE non-captures: " << std::endl;
	std::cout << (notCaptureFilter.stats() && seeFilterNeutral.stats()).ToString() << std::endl;

	std::cout << "-SEE captures: " << std::endl;
	std::cout << (isCaptureFilter.stats() && seeFilterNeg.stats()).ToString() << std::endl;

	std::cout << "-SEE non-captures: " << std::endl;
	std::cout << (notCaptureFilter.stats() && seeFilterNeg.stats()).ToString() << std::endl;

	std::cout << std::endl;

	std::cout << "Highest SEE captures:" << std::endl;
	auto highestSeeCaptures = isCaptureFilter.stats() && highestSeeFilter.stats();
	std::cout << (highestSeeCaptures).ToString() << std::endl;

	std::cout << "Non-highest +SEE captures:" << std::endl;
	std::cout << (posSeeCaptures ^ highestSeeCaptures).ToString() << std::endl;

	std::cout << std::endl;

	std::cout << "Piece Types (+SEE):" << std::endl;
	for (auto &ptf : pieceTypeFilters)
	{
		auto seePieceType = ptf.second.stats() && seeFilterPos.stats();
		std::cout << PieceTypeToChar(ptf.first) << " (opening): " <<
			(seePieceType && isOpening.stats()).ToString() << std::endl;
		std::cout << PieceTypeToChar(ptf.first) << " (end):     " <<
			(seePieceType && isEndGame.stats()).ToString() << std::endl;
	}

	std::cout << std::endl;

	std::cout << "Piece Types (=SEE):" << std::endl;
	for (auto &ptf : pieceTypeFilters)
	{
		auto seePieceType = ptf.second.stats() && seeFilterNeutral.stats();
		std::cout << PieceTypeToChar(ptf.first) << " (opening): " <<
			(seePieceType && isOpening.stats()).ToString() << std::endl;
		std::cout << PieceTypeToChar(ptf.first) << " (end):     " <<
			(seePieceType && isEndGame.stats()).ToString() << std::endl;
	}

	std::cout << std::endl;

	std::cout << "Piece Types (-SEE):" << std::endl;
	for (auto &ptf : pieceTypeFilters)
	{
		auto seePieceType = ptf.second.stats() && seeFilterNeg.stats();
		std::cout << PieceTypeToChar(ptf.first) << " (opening): " <<
			(seePieceType && isOpening.stats()).ToString() << std::endl;
		std::cout << PieceTypeToChar(ptf.first) << " (end):     " <<
			(seePieceType && isEndGame.stats()).ToString() << std::endl;
	}

	std::cout << std::endl;

	std::cout << "Queen promotions: " << isQPromotionFilter.stats().ToString() << std::endl;

	std::cout << std::endl;

	std::cout << "Under-promotions: " << (isPromotionFilter.stats() ^ isQPromotionFilter.stats()).ToString() << std::endl;

	std::cout << std::endl;

	std::cout << "Escapes: " << isEscape.stats().ToString() << std::endl;

//	std::cout << "Safe promotions:" << std::endl;
//	std::cout << (isPromotionFilter.stats() && SEEPositiveOrNeutral).ToString() << std::endl;

//	std::cout << std::endl;

//	std::cout << "Unsafe promotions:" << std::endl;
//	std::cout << (isPromotionFilter.stats() && seeFilterNeg.stats()).ToString() << std::endl;

//	std::cout << std::endl;

//	std::cout << "Safe under-promotions:" << std::endl;
//	auto underPromotions = isPromotionFilter.stats() ^ isQPromotionFilter.stats();
//	std::cout << (underPromotions && SEEPositiveOrNeutral).ToString() << std::endl;
}

}

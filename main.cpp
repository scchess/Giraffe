#include <iostream>

#include "magic_moves.h"
#include "board_consts.h"
#include "move.h"
#include "board.h"

void Initialize()
{
	initmagicmoves();
	BoardConstsInit();
}

int main(int argc, char **argv)
{
#ifdef DEBUG
	std::cout << "# Running in debug mode" << std::endl;
#else
	std::cout << "# Running in release mode" << std::endl;
#endif

	Initialize();

#if 1
	DebugRunPerftTests();

	return 0;
#else
	std::string fen;
	std::getline(std::cin, fen);

	Board b(fen);

	while (true)
	{

		std::cout << b.PrintBoard() << std::endl;

		MoveList ml;

		/*
		b.RemovePiece(E1);
		b.PlacePiece(B6, WK);
		b.PlacePiece(B5, WB);
		b.RemovePiece(D1);
		b.PlacePiece(D4, WQ);
		b.PlacePiece(F5, WN);
		*/

		std::cout << "all:" << std::endl;
		b.GenerateAllMoves<Board::ALL>(ml);

		for (size_t i = 0; i < ml.GetSize(); ++i)
		{
			std::cout << i << ": " << b.MoveToAlg(ml[i]) << std::endl;
		}

		int moveChoice;
		std::cin >> moveChoice;

		b.ApplyMove(ml[moveChoice]);
		std::cout << b.PrintBoard() << std::endl;
	}
#endif
}

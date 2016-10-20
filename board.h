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

#ifndef BOARD_H
#define BOARD_H

#include <array>
#include <functional>
#include <string>
#include <sstream>
#include <cstddef>

#include "types.h"
#include "board_consts.h"
#include "move.h"
#include "bit_ops.h"

const static std::string DEFAULT_POSITION_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// these definitions are used as indices for the board description arrays
// all aspects of a position are in the 2 arrays (one for bitboards, one for byte fields, including mailbox representation) for ease of undo-ing moves

// the first array starts with bitboards for each of the 12 piece types
// 0x0 to 0xd are used for storing piece bitboards
const static ptrdiff_t WHITE_OCCUPIED = 0x6;
const static ptrdiff_t BLACK_OCCUPIED = 0xe;
const static ptrdiff_t EN_PASS_SQUARE = 0x10; // stored as a bitboard since we have 64 bits anyways
const static ptrdiff_t BOARD_HASH = 0x11;

const static ptrdiff_t HASH = 0x12;

const static ptrdiff_t BOARD_DESC_BB_SIZE = 0x13;

// the second array starts with a mailbox representation of the board, from 0x0 to 0x3F
// we also keep track of misc things in this array
// castling rights
const static ptrdiff_t W_SHORT_CASTLE = 0x40;
const static ptrdiff_t W_LONG_CASTLE = 0x41;
const static ptrdiff_t B_SHORT_CASTLE = 0x42;
const static ptrdiff_t B_LONG_CASTLE = 0x43;

const static ptrdiff_t SIDE_TO_MOVE = 0x44;

// half moves clock will overflow at 256 here, but hopefully that won't happen very often
// the engine shouldn't crash regardless (just misevaluate draw)
const static ptrdiff_t HALF_MOVES_CLOCK = 0x45; // number of half moves since last irreversible move (for 50 moves detection)

const static ptrdiff_t IN_CHECK = 0x46; // whether the moving side is in check (this is updated on each board change, so we don't have to recompute many times)

const static ptrdiff_t BOARD_DESC_U8_SIZE = 0x47;

class Board
{
public:
	enum MOVE_TYPES
	{
		QUIET,
		VIOLENT,
		ALL
	};

	enum GameStatus
	{
		WHITE_WINS,
		BLACK_WINS,
		STALEMATE,
		INSUFFICIENT_MATERIAL,
		ONGOING
	};

	enum MoveFormat
	{
		ALGEBRAIC,
		SAN
	};

	struct CheckInfo
	{
		// this struct contains things that can be precomputed once per position, and makes checking all the moves easier
		bool opponentRQOnSameX = false;
		bool opponentRQOnSameY = false;
		bool opponentBQOnSameDiag0 = false;
		bool opponentBQOnSameDiag1 = false;
	};

	// these are features of the board that change slowly (used in eval caching)
	struct SlowFeatures
	{
		Color stm;
		Square wk;
		Square bk;
		uint64_t wp;
		uint64_t bp;

		uint8_t pieceCounts[NUM_PIECETYPES];

		size_t Hash()
		{
			std::stringstream ss;

			ss << stm << ' ' << wk << ' ' << bk << ' ' << wp << ' ' << bp << ' ';

			for (uint32_t i = 0; i < NUM_PIECETYPES; ++i)
			{
				ss << pieceCounts[i] << ' ';
			}

			std::hash<std::string> hashFcn;

			return hashFcn(ss.str());
		}
	};

	using MoveCountMap = std::array<size_t, 64>;

	typedef FixedVector<std::pair<uint8_t, uint64_t>, 7> UndoListBB; // list of bitboards to revert on undo
	// 6 maximum bitboards (black occupied, white occupied, source piece type, captured piece type, promotion/castling piece type, en passant, hash)

	typedef FixedVector<std::pair<uint8_t, uint8_t>, 8> UndoListU8;
	// For en passant (en passants cannot result in promotion, or reducing castling rights):
	// half moves, in check, source square, destination square, en pass captured square - 5
	// For promotions (capture and non-capture, cannot reduce castling rights):
	// half moves, in check, source square, destination square - 4
	// For regular moves and captures
	// half moves, in check, source square, destination square, changing at most 2 castling rights - 6
	// For castling
	// half moves, in check, king from, king to, rook from, rook to, 2 castling rights - 8


	Board(const std::string &fen);
	Board() : Board(DEFAULT_POSITION_FEN) {}
	~Board() {}

	void RemovePiece(Square sq);
	void PlacePiece(Square sq, PieceType pt);

	template <MOVE_TYPES MT> void GenerateAllLegalMoves(MoveList &moveList);

	// debug function to check consistency between occupied bitboards, piece bitboards, MB, and castling rights
	void CheckBoardConsistency();

	std::string GetFen(bool omitMoveNums = false) const;

	std::string PrintBoard() const;

	bool InCheck() const { return m_boardDescU8[IN_CHECK]; }

	// returns whether the move is legal (if not, the move is reverted)
	bool ApplyMove(Move mv);

	CheckInfo ComputeCheckInfo() const;

	// same as ApplyMove, but doesn't apply the move
	// it also uses a few shortcuts to do the check faster
	bool CheckLegal(const CheckInfo &ci, Move mv);

	void UndoMove();

	std::string MoveToAlg(Move mv, MoveFormat mf = ALGEBRAIC);

	std::string PVToStr(std::vector<Move> &pv) const;

	bool operator==(const Board &other);

	uint64_t GetPieceTypeBitboard(PieceType pt) const { return m_boardDescBB[pt]; }

	template <Color COLOR>
	uint64_t GetOccupiedBitboard() const
	{ return (COLOR == WHITE) ? m_boardDescBB[WHITE_OCCUPIED] : m_boardDescBB[BLACK_OCCUPIED]; }

	Color GetSideToMove() const { return m_boardDescU8[SIDE_TO_MOVE]; }

	PieceType GetPieceAtSquare(Square sq) const { return m_boardDescU8[sq]; }

	Move ParseMove(std::string str);

	// how many moves can be undone from the current position
	int32_t PossibleUndo() { return m_undoStackBB.GetSize(); }

	uint64_t GetHash() const { return m_boardDescBB[HASH]; }

	// is it probable that this position is zugzwang (used in null move)
	bool IsZugzwangProbable();

	// position must not be in check, otherwise behaviour is undefined
	// null moves are recorded in the undo stacks, and can be undone using undo
	void MakeNullMove();

	// check whether the move is legal
	// the move must be a legal move in SOME position (for example, no king promotions, or knights moving like a pawn)
	bool CheckPseudoLegal(Move mv);

	bool IsViolent(Move mv);

	// whether the moving side has pawn on 7th rank or not
	bool HasPawnOn7th();

	// get the largest piece type of the opponent, in white's type
	PieceType GetOpponentLargestPieceType();

	// if this position has appeared 3 times before
	// note, there is an intentional bug here -
	// en passant status only counts if there is actually a pawn to do the capture
	// we count it anyways since that's extremely rare, we never claim draws
	// (only offer, which becomes a claim if the GUI thinks the draw is claimable)
	// in the case that the "claim" is incorrect, we will simply play on (after offering a draw)
	bool Is3Fold();

	bool Is50Moves() { return m_boardDescU8[HALF_MOVES_CLOCK] >= 100; }

	// look for a repetition in the last numMoves
	// this is used in the search
	// we don't look through the whole history because that can be very slow in long games
	bool Is2Fold(size_t numMoves);

	bool IsEpAvailable() const { return m_boardDescBB[EN_PASS_SQUARE] != 0; }
	Square GetEpSquare() const { return BitScanForward(m_boardDescBB[EN_PASS_SQUARE]); }

	// in relaxed mode, we include material configurations that are not drawn by rule, but are
	// effectively drawn (helpmate situations)
	bool HasInsufficientMaterial(bool relaxed = true) const;

	GameStatus GetGameStatus();

	/*
		SEE helpers
		- Highly efficient limited ApplyMove/UndoMove for SEE only
		- Move must be a legal regular capture
		- No en passant, castling, or non-capture moves
		- Move legality is not checked, and king is allowed to be captured
		- Same number of UndoMoveSee() must be called before
			any other function can be called (board is in an intentionally corrupted
			state while in SEE mode)
		- ApplyMoveSee returns the captured piecetype
		- ResetSee resets SEE status
	*/
	void ResetSee() { m_seeLastWhitePT = WP; m_seeLastBlackPT = WP; m_seeTotalOccupancy = m_boardDescBB[WHITE_OCCUPIED] | m_boardDescBB[BLACK_OCCUPIED]; }
	PieceType ApplyMoveSee(PieceType pt, Square from, Square to);
	bool IsSeeEligible(Move mv);
	void UndoMoveSee();
	bool GenerateSmallestCaptureSee(PieceType &pt, Square &from, Square to); // to doesn't need to be returned, because it's the target square

	// undefined behaviour if move is not violent
	PieceType GetCapturedPieceType(Move violentMove);

	uint64_t SpeculateHashAfterMove(Move mv);

	size_t GetPieceCount(PieceType pt) const { return PopCount(m_boardDescBB[pt]); }

	bool HasCastlingRight(uint32_t right) const { return m_boardDescU8[right]; }

	// get the position of any piece of piece type (this is mostly used for kings)
	size_t GetFirstPiecePos(PieceType pt) const { return BitScanForward(m_boardDescBB[pt]); }

	template <PieceType PT>
	uint64_t GetAttackers(Square sq) const;

	void ApplyVariation(const std::vector<Move> &moves);

	// get the least valuable attacker
	// all piece types are white
	void ComputeLeastValuableAttackers(PieceType attackers[64], uint8_t numAttackers[64], Color side);

	void ComputeMoveCountMap(MoveCountMap &m)
	{
		for (auto &x : m)
		{
			x = 0;
		}

		MoveList ml;
		GenerateAllLegalMoves<ALL>(ml);
		for (const auto &move : ml)
		{
			++m[GetFromSquare(move)];
		}
	}

	// 0 = last move, 1 = last move - 1, etc
	Optional<Move> GetMoveFromLast(int32_t n);

	void GetSlowFeatures(SlowFeatures &sf)
	{
		sf.stm = GetSideToMove();
		sf.wk = BitScanForward(m_boardDescBB[WK]);
		sf.bk = BitScanForward(m_boardDescBB[BK]);
		sf.wp = m_boardDescBB[WP];
		sf.bp = m_boardDescBB[BP];

		for (uint32_t i = 0; i < NUM_PIECETYPES; ++i)
		{
			sf.pieceCounts[i] = PopCount(m_boardDescBB[PIECE_TYPE_INDICES[i]]);
		}
	}

	bool IsChecking(Move mv)
	{
		// TODO: re-implement more efficiently if this becomes a bottleneck
		ApplyMove(mv);
		bool ret = InCheck();
		UndoMove();

		return ret;
	}

	int32_t GetHalfMoveCount() const
	{
		return static_cast<int32_t>(m_undoStackBB.GetSize());
	}

	Board GetMirroredPosition() const;

private:
	template <MOVE_TYPES MT> void GenerateAllPseudoLegalMoves_(MoveList &moveList) const;
	template <MOVE_TYPES MT> void GenerateKingMoves_(Color color, MoveList &moveList) const;
	template <MOVE_TYPES MT> void GenerateQueenMoves_(Color color, MoveList &moveList) const;
	template <MOVE_TYPES MT> void GenerateBishopMoves_(Color color, MoveList &moveList) const;
	template <MOVE_TYPES MT> void GenerateKnightMoves_(Color color, MoveList &moveList) const;
	template <MOVE_TYPES MT> void GenerateRookMoves_(Color color, MoveList &moveList) const;

	// non-quiet only generates captures and promotion to queen
	// quiet only generates non-captures and under-promotions (including captures that result in under-promotion)
	template <MOVE_TYPES MT> void GeneratePawnMoves_(Color color, MoveList &moveList) const;

	bool IsUnderAttack_(Square sq) const;
	void UpdateInCheck_();

	void UpdateHashFull_();

	uint64_t m_boardDescBB[BOARD_DESC_BB_SIZE];

	// yes, we are using uint64_t to store these u8 values
	// testing shows that this is the fastest
	// testing also shows that using uint8_t in the undo list is the fastest, despite the size mismatch
	uint64_t m_boardDescU8[BOARD_DESC_U8_SIZE];

	GrowableStack<UndoListBB> m_undoStackBB;
	GrowableStack<UndoListU8> m_undoStackU8;

	// stack of hashes, for detecting repetition
	GrowableStack<uint64_t> m_hashStack;

	GrowableStack<Move> m_moveStack;

	// both these fields are stored as white piece types
	void UpdateseeLastPT_(PieceType lastPT) { if (m_boardDescU8[SIDE_TO_MOVE] == WHITE) m_seeLastWhitePT = lastPT; else m_seeLastBlackPT = lastPT; }
	PieceType m_seeLastWhitePT;
	PieceType m_seeLastBlackPT;
	uint64_t m_seeTotalOccupancy;
};

uint64_t DebugPerft(Board &b, uint32_t depth);

// same as perft, but also tries a null move on all non-in-check positions
uint64_t DebugPerftWithNull(Board &b, uint32_t depth);

void DebugRunPerftTests();

void DebugRunSANTests();

#endif // BOARD_H

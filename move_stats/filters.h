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

#ifndef FILTERS_H
#define FILTERS_H

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

#include "../board.h"
#include "../types.h"
#include "../see.h"

namespace MoveStats
{

inline uint64_t Hash(Board &b, Move mv) { return b.GetHash() ^ GetMoveHash(mv); }

struct FilterStats
{
	struct MatchEntry
	{
		uint64_t hash;
		bool isBest;
		float scalingFactor;

		bool operator<(const MatchEntry &other)
		{
			return hash < other.hash;
		}

		bool operator==(const MatchEntry &other)
		{
			return hash == other.hash;
		}
	};

	std::vector<MatchEntry> matches;
	bool sorted = false;

	std::string ToString()
	{
		if (matches.size() == 0)
		{
			return "no match";
		}

		uint64_t bestCount = 0;
		float totalScalingFactor = 0;

		for (const auto &x : matches)
		{
			if (x.isBest)
			{
				++bestCount;
			}

			totalScalingFactor += x.scalingFactor;
		}

		std::stringstream ss;
		float scaledProb = bestCount / totalScalingFactor;
		//ss << "best: " << bestCount << "\tlegal: " << matches.size() << "\tscaled: " << scaledProb;
		ss << scaledProb;
		return ss.str();
	}
};

inline FilterStats operator&&(FilterStats &a, FilterStats &b)
{
	FilterStats newStats;

	if (!a.sorted)
	{
		std::sort(a.matches.begin(), a.matches.end());
		a.sorted = true;
	}

	if (!b.sorted)
	{
		std::sort(b.matches.begin(), b.matches.end());
		b.sorted = true;
	}

	std::set_intersection(a.matches.begin(), a.matches.end(),
						  b.matches.begin(), b.matches.end(),
						  std::back_inserter(newStats.matches));

	newStats.sorted = true;

	return newStats;
}

inline FilterStats operator||(FilterStats &a, FilterStats &b)
{
	FilterStats newStats;

	if (!a.sorted)
	{
		std::sort(a.matches.begin(), a.matches.end());
		a.sorted = true;
	}

	if (!b.sorted)
	{
		std::sort(b.matches.begin(), b.matches.end());
		b.sorted = true;
	}

	std::set_union(a.matches.begin(), a.matches.end(),
				   b.matches.begin(), b.matches.end(),
				   std::back_inserter(newStats.matches));

	auto last = std::unique(newStats.matches.begin(), newStats.matches.end());
	newStats.matches.erase(last, newStats.matches.end());
	newStats.sorted = true;

	return newStats;
}

inline FilterStats operator^(FilterStats &a, FilterStats &b)
{
	FilterStats newStats;

	if (!a.sorted)
	{
		std::sort(a.matches.begin(), a.matches.end());
		a.sorted = true;
	}

	if (!b.sorted)
	{
		std::sort(b.matches.begin(), b.matches.end());
		b.sorted = true;
	}

	std::set_difference(a.matches.begin(), a.matches.end(),
						b.matches.begin(), b.matches.end(),
						std::back_inserter(newStats.matches));
	newStats.sorted = true;

	return newStats;
}

class FilterIf
{
public:
	/* Precompute things that are common for moves on the same board */
	virtual void Precompute(Board &/*board*/, MoveList &/*moveList*/) {}

	/* Run the filter without updating stats */
	virtual bool Match(Board &board, Move move) = 0;

	FilterStats &stats() { return m_stats; }

	/* Run the filter and update stats */
	void RunFilter(Board &board, Move bestMove);

private:
	FilterStats m_stats;
};

class PieceTypeFilter : public FilterIf
{
public:
	PieceTypeFilter() {}
	PieceTypeFilter(PieceType pt) : m_pt(StripColor(pt)) {}

	bool Match(Board &/*board*/, Move move) override
	{
		return StripColor(GetPieceType(move)) == m_pt;
	}

private:
	PieceType m_pt;
};

class FromSquareFilter : public FilterIf
{
public:
	FromSquareFilter(Square sq) : m_sq(sq) {}

	bool Match(Board &/*board*/, Move move) override
	{
		return GetFromSquare(move) == m_sq;
	}

private:
	Square m_sq;
};

class ToSquareFilter : public FilterIf
{
public:
	ToSquareFilter(Square sq) : m_sq(sq) {}

	bool Match(Board &/*board*/, Move move) override
	{
		return GetToSquare(move) == m_sq;
	}

private:
	Square m_sq;
};

class SEEFilter : public FilterIf
{
public:
	enum Mode
	{
		POS, POS_OR_NEUTRAL, NEUTRAL, NEUTRAL_OR_NEG, NEG
	};

	SEEFilter(Mode mode) : m_mode(mode) {}

	bool Match(Board &board, Move move) override
	{
		Score see = SEE::StaticExchangeEvaluation(board, move);

		if (see > 0)
		{
			return m_mode == POS || m_mode == POS_OR_NEUTRAL;
		}
		else if (see == 0)
		{
			return m_mode == POS_OR_NEUTRAL || m_mode == NEUTRAL || m_mode == NEUTRAL_OR_NEG;
		}
		else
		{
			return m_mode == NEUTRAL_OR_NEG || m_mode == NEG;
		}
	}

private:
	Mode m_mode;
};

class HighestSEEFilter : public FilterIf
{
public:
	void Precompute(Board &board, MoveList &moveList) override
	{
		Score highestSee = std::numeric_limits<Score>::lowest();

		for (const auto &move : moveList)
		{
			Score see = SEE::StaticExchangeEvaluation(board, move);
			if (see > highestSee)
			{
				highestSee = see;
				m_highestMoves.clear();
				m_highestMoves.insert(move);
			}
			else if (see == highestSee)
			{
				m_highestMoves.insert(move);
			}
		}
	}

	bool Match(Board &/*board*/, Move move) override
	{
		if (m_highestMoves.size() > 1)
		{
			return false;
		}

		return m_highestMoves.find(move) != m_highestMoves.end();
	}

private:
	std::set<Move> m_highestMoves;
};

class PromotionFilter : public FilterIf
{
public:
	bool Match(Board &/*board*/, Move move) override
	{
		return IsPromotion(move);
	}
};

class PromotionTypeFilter : public FilterIf
{
public:
	PromotionTypeFilter(PieceType pt) : m_pt(StripColor(pt)) {}

	bool Match(Board &/*board*/, Move move) override
	{
		return IsPromotion(move) && StripColor(GetPromoType(move)) == m_pt;
	}

private:
	PieceType m_pt;
};

class IsCaptureFilter : public FilterIf
{
public:
	IsCaptureFilter() {}

	bool Match(Board &board, Move move) override
	{
		return board.IsViolent(move);
	}
};

/* Filter by piece count not including kings and pawns */
class GamePhaseFilter : public FilterIf
{
public:
	enum Mode {
		MORE_THAN_OR_EQUAL,
		LESS_THAN
	};

	GamePhaseFilter(size_t totalPiecesCount, Mode mode) : m_pc(totalPiecesCount), m_mode(mode) {}

	bool Match(Board &board, Move /*move*/) override
	{
		size_t count = board.GetPieceCount(WQ) + board.GetPieceCount(WR) + board.GetPieceCount(WB) + board.GetPieceCount(WN) +
				board.GetPieceCount(BQ) + board.GetPieceCount(BR) + board.GetPieceCount(BB) + board.GetPieceCount(BN);

		if (m_mode == MORE_THAN_OR_EQUAL)
		{
			return count >= m_pc;
		}
		else
		{
			return count < m_pc;
		}
	}

private:
	size_t m_pc;
	Mode m_mode;
};

class EscapeFilter : public FilterIf
{
public:
	bool Match(Board &board, Move move) override
	{
		return SEE::NMStaticExchangeEvaluation(board, move) > 0;
	}
};

// Logical filters
class FilterNot : public FilterIf
{
public:
	FilterNot(FilterIf *filter) : m_filter(filter) {}

	bool Match(Board &board, Move move) override
	{
		return !m_filter->Match(board, move);
	}

private:
	FilterIf *m_filter;
};

}

#endif // FILTERS_H

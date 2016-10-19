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

#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "types.h"
#include "board.h"
#include "see.h"

#include <limits>

// add small offsets to prevent overflow/underflow on adding/subtracting 1 (eg. for PV search)
const static Score SCORE_MAX = std::numeric_limits<Score>::max() - 1000;
const static Score SCORE_MIN = std::numeric_limits<Score>::lowest() + 1000;

class EvaluatorIface
{
public:
	constexpr static float EvalFullScale = 10000.0f;

	virtual bool IsANNEval() { return false; }

	// return score for side to move
	virtual Score EvaluateForSTM(Board &b, Score lowerBound = SCORE_MIN, Score upperBound = SCORE_MAX)
	{
		if (b.GetSideToMove() == WHITE)
		{
			return EvaluateForWhiteImpl(b, lowerBound, upperBound);
		}
		else
		{
			return -EvaluateForWhiteImpl(b, -upperBound, -lowerBound);
		}
	}

	virtual Score EvaluateForWhite(Board &b, Score lowerBound = SCORE_MIN, Score upperBound = SCORE_MAX)
	{
		return EvaluateForWhiteImpl(b, lowerBound, upperBound);
	}

	virtual float UnScale(float x)
	{
		float ret = x / EvalFullScale;

		ret = std::max(ret, -1.0f);
		ret = std::min(ret, 1.0f);

		return ret;
	}

	// this is the only function evaluators need to implement
	virtual Score EvaluateForWhiteImpl(Board &b, Score lowerBound, Score upperBound) = 0;

	// this allows evaluators to evaluate multiple positions at once
	// default implementation does it one at a time
	virtual void BatchEvaluateForWhiteImpl(std::vector<Board> &positions, std::vector<Score> &results, Score lowerBound, Score upperBound)
	{
		results.resize(positions.size());

		for (size_t i = 0; i < positions.size(); ++i)
		{
			results[i] = EvaluateForWhiteImpl(positions[i], lowerBound, upperBound);
		}
	}

	// this is optional
	virtual void PrintDiag(Board &/*board*/) {}

	virtual ~EvaluatorIface() {}
};

#endif // EVALUATOR_H

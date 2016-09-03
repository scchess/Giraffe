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

#ifndef ANN_BUILDER_H
#define ANN_BUILDER_H

#include "Eigen/Core"

#include "ann.h"

namespace AnnBuilder
{

EvalNet BuildEvalNet(int64_t inputDims, int64_t outputDims, bool smallNet);

MoveEvalNet BuildMoveEvalNet(int64_t inputDims, int64_t outputDims);

}

#endif // ANN_BUILDER_H

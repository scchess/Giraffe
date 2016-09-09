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

#ifndef LEARN_H
#define LEARN_H

#include <string>
#include <fstream>

#include <iostream>

#include "Eigen/Core"

namespace Learn
{

const static int64_t NumIterations = 1000000;
const static float TDLambda = 0.7f; // this is discount due to credit assignment uncertainty
const static float AbsLambda = 0.995f; // this is discount to encourage progress, and account for the snowball effect
const static int64_t HalfMovesToMake = 32;
const static size_t PositionsFirstIteration = 100000;
const static int64_t PositionsEarlyIterations = 100000;
const static int64_t NumEarlyIteartions = 10;
const static int64_t PositionsPerIteration = 1000000;
const static int64_t SgdBatchSize = 1024;
const static int64_t SgdEpochs = 10;
const static int64_t SearchNodeBudget = 256;
const static int64_t EvaluatorSerializeInterval = 1;
const static int64_t IterationPrintInterval = 1;
const static std::string TrainingLogFileName = "training.log";

void TDL(const std::string &positionsFilename);

}

#endif // LEARN_H

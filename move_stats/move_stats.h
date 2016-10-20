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

#ifndef MOVE_STATS_H
#define MOVE_STATS_H

#include <string>

#include "filters.h"
#include "../board.h"
#include "../types.h"

namespace MoveStats
{

void ProcessStats(const std::string &filename);

}
#endif // MOVE_STATS_H

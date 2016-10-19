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

#ifndef FEATURES_CONV_H
#define FEATURES_CONV_H

#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <utility>

#include "Eigen/Dense"

#include "ann.h"
#include "board.h"
#include "types.h"
#include "containers.h"
#include "move.h"
#include "consts.h"

namespace FeaturesConv
{

struct FeatureDescription
{
	enum FeatureType
	{
		FeatureType_global, // global features are things like side to move, and material counts, and piece lists
		FeatureType_pos // property of a square
	};

	FeatureType featureType;

	// fields for global and pos features
	int32_t group;

	// fields for pos features
	Square sq;

	std::string ToString() const
	{
		std::stringstream ret;

		switch (featureType)
		{
		case FeatureType_global:
			ret << "GLOBAL ";
			ret << group << ' ';
			break;
		case FeatureType_pos:
			ret << "POS_GN ";
			ret << sq;
			break;
		default:
			assert(false);
		}

		return ret.str();
	}
};

using GroupAllocations = std::vector<std::pair<int64_t /* size */, float /* reduction factor */>>;

// convert to NN input format
// T can either be float (to get actual values) or
// FeatureDescription (to get feature descriptions)
template <typename T>
void ConvertBoardToNN(Board &board, std::vector<T> &ret);

inline int64_t GetNumFeatures()
{
	Board b;

	std::vector<FeaturesConv::FeatureDescription> ret;
	FeaturesConv::ConvertBoardToNN(b, ret);

	return static_cast<int64_t>(ret.size());
}

// Divide features into groups (this doesn't have to match actual groups in features)
inline GroupAllocations GetBoardGroupAllocations()
{
	Board b;

	std::vector<FeaturesConv::FeatureDescription> fds;
	FeaturesConv::ConvertBoardToNN(b, fds);

	GroupAllocations ret;

	int current_group = 0; /* we know the first feature will be in the global0 group */
	int current_group_size = 0;

	for (const auto &fd : fds)
	{
		int group = 0;

		if (fd.group == 0)
		{
			group = 0;
		}
		else if (fd.group == 1)
		{
			/* pawn group */
			group = 1;
		}
		else if (fd.featureType != FeatureDescription::FeatureType_pos)
		{
			group = 2; // other globals
		}
		else
		{
			group = 3; // square features
		}

		if (group == current_group)
		{
			++current_group_size;
		}
		else
		{
			float reduction_factor = 0.25f;

			if (current_group == 0)
			{
				// first global group is very important
				reduction_factor = 1.0f;
			}
			else if (current_group == 1)
			{
				// the pawn group is huge
				reduction_factor = 0.2f;
			}
			else if (current_group == 3)
			{
				// there are many square features
				reduction_factor = 0.15f;
			}

			ret.push_back(std::pair<int64_t, float>(current_group_size, reduction_factor));
			current_group = group;
			current_group_size = 1;
		}
	}

	float reduction_factor = 0.25f;

	if (current_group == 0)
	{
		// first global group is very important
		reduction_factor = 1.0f;
	}
	else if (current_group == 1)
	{
		// the pawn group is huge
		reduction_factor = 0.2f;
	}
	else if (current_group == 3)
	{
		// there are many square features
		reduction_factor = 0.15f;
	}

	// last group
	ret.push_back(std::pair<int64_t, float>(current_group_size, reduction_factor));

	return ret;
}

// additional info for conversion
struct ConvertMovesInfo
{
	std::vector<Score> see;
	std::vector<Score> nmSee; // SEE of the source square
};

// convert a list of moves to NN input format
void ConvertMovesToNN(
	Board &board,
	ConvertMovesInfo &convInfo,
	MoveList &ml,
	NNMatrixRM &ret);

// because of the way we convert a move list at a time, it's not possible to do the same thing with
// ConvertBoardToNN (using templatized functions to perform feature description extraction)
// so we need a separate function
void GetMovesFeatureDescriptions(std::vector<FeaturesConv::FeatureDescription> &fds);

inline int64_t GetMoveNumFeatures()
{
	std::vector<FeaturesConv::FeatureDescription> fds;
	GetMovesFeatureDescriptions(fds);
	return static_cast<int64_t>(fds.size());
}

}

#endif // FEATURES_CONV_H

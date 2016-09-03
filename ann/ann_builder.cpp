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

#include "ann_builder.h"

#include <iostream>
#include <fstream>
#include <random>
#include <thread>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <queue>
#include <algorithm>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <omp.h>

#include <cmath>

#include <sys/types.h>
#include <sys/stat.h>

#ifndef _WIN32
#include <sys/mman.h>
#endif

#include <fcntl.h>

#include "ann.h"
#include "types.h"
#include "random_device.h"
#include "features_conv.h"
#include "consts.h"

namespace
{
typedef std::vector<int32_t> Group;

std::vector<std::tuple<size_t, size_t>> GetCombinations(size_t m /* for combinations of 0-10, use m = 11 */)
{
	std::vector<std::tuple<size_t, size_t>> ret;

	for (size_t elem0 = 0; elem0 < m; ++elem0)
	{
		for (size_t elem1 = 0; elem1 < elem0; ++elem1)
		{
			ret.push_back(std::make_tuple(elem0, elem1));
		}
	}

	return ret;
}

struct LayerDescription
{
	size_t layerSize;
	std::vector<Eigen::Triplet<float> > connections;

	LayerDescription() : layerSize(0) {}
};

void AddSingleNodesGroup(
	LayerDescription &layerDescription,
	const Group &groupIn,
	Group &groupOut,
	float nodeCountMultiplier
	)
{
	size_t nodesInGroup = groupIn.size();
	size_t nodesForThisGroup = ceil(nodesInGroup * nodeCountMultiplier);

	groupOut.clear();

	for (size_t i = 0; i < nodesForThisGroup; ++i)
	{
		for (auto feature : groupIn)
		{
			layerDescription.connections.push_back(Eigen::Triplet<float>(feature, layerDescription.layerSize, 1.0f));
		}

		groupOut.push_back(layerDescription.layerSize);

		++layerDescription.layerSize;
	}
}

void DebugPrintGroups(const std::vector<Group> &groups)
{
	std::cout << "Groups:" << std::endl;
	size_t groupNum = 0;
	for (auto group : groups)
	{
		std::cout << groupNum << " (" << group.size() << "): ";

		for (auto feature : group)
		{
			std::cout << feature << ' ';
		}

		std::cout << std::endl;

		++groupNum;
	}
}

void AnalyzeFeatureDescriptions(const std::vector<FeaturesConv::FeatureDescription> &featureDescriptions,
								Group &globalGroup, /* global group does not include group 0! */
								Group &squareGroup,
								Group &group0)
{
	// first we make global feature groups
	for (size_t featureNum = 0; featureNum < featureDescriptions.size(); ++featureNum)
	{
		auto &fd = featureDescriptions[featureNum];

		if (fd.featureType == FeaturesConv::FeatureDescription::FeatureType_global)
		{
			if (fd.group == 0)
			{
				group0.push_back(featureNum);
			}
			else
			{
				globalGroup.push_back(featureNum);
			}
		}
		else if (fd.featureType == FeaturesConv::FeatureDescription::FeatureType_pos)
		{
			squareGroup.push_back(featureNum);
		}
	}

	//assert(squareGroup.size() == (2*64));
	assert(group0.size() > 5 && group0.size() < 40);
}

} // namespace

namespace AnnBuilder
{

EvalNet BuildEvalNet(int64_t inputDims, int64_t outputDims, bool smallNet)
{
	std::vector<size_t> layerSizes;
	std::vector<std::vector<Eigen::Triplet<float> > > connMatrices;

	Group globalGroup;
	Group squareGroup;
	Group group0;

	// get feature descriptions
	std::vector<FeaturesConv::FeatureDescription> featureDescriptions;
	Board dummyBoard;
	FeaturesConv::ConvertBoardToNN(dummyBoard, featureDescriptions);

	AnalyzeFeatureDescriptions(featureDescriptions,
									globalGroup,
									squareGroup,
									group0);

	if (!smallNet)
	{
		LayerDescription layer0;

		Group layer0Group0;
		Group layer0GlobalGroup;
		Group layer0SquareGroup;

		// first we add the mixed global group
		AddSingleNodesGroup(layer0, globalGroup, layer0GlobalGroup, 0.2f);

		// mixed square group
		AddSingleNodesGroup(layer0, squareGroup, layer0SquareGroup, 0.2f);

		// pass through group 0 (this contains game phase information)
		AddSingleNodesGroup(layer0, group0, layer0Group0, 1.0f);

		layerSizes.push_back(layer0.layerSize);
		connMatrices.push_back(layer0.connections);

		// in the second layer, we just fully connect everything
		layerSizes.push_back(BoardSignatureSize);
		connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

		// fully connected output layer
		connMatrices.push_back(std::vector<Eigen::Triplet<float> >());
	}
	else
	{
		LayerDescription layer0;

		Group layer0Group0;
		Group layer0GlobalGroup;
		Group layer0SquareGroup;

		// first we add the mixed global group
		AddSingleNodesGroup(layer0, globalGroup, layer0GlobalGroup, 0.1f);

		// mixed square group
		AddSingleNodesGroup(layer0, squareGroup, layer0SquareGroup, 0.1f);

		// pass through group 0 (this contains game phase information)
		AddSingleNodesGroup(layer0, group0, layer0Group0, 1.0f);

		layerSizes.push_back(layer0.layerSize);
		connMatrices.push_back(layer0.connections);

		// in the second layer, we just fully connect everything
		layerSizes.push_back(BoardSignatureSize);
		connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

		// fully connected output layer
		connMatrices.push_back(std::vector<Eigen::Triplet<float> >());
	}

	return EvalNet(inputDims, outputDims, layerSizes, connMatrices);
}

MoveEvalNet BuildMoveEvalNet(int64_t inputDims, int64_t outputDims)
{
	std::vector<size_t> layerSizes;
	std::vector<std::vector<Eigen::Triplet<float> > > connMatrices;

	Group globalGroup;
	Group squareGroup;
	Group group0;

	// get feature descriptions
	std::vector<FeaturesConv::FeatureDescription> featureDescriptions;
	GetMovesFeatureDescriptions(featureDescriptions);

	AnalyzeFeatureDescriptions(featureDescriptions,
									globalGroup,
									squareGroup,
									group0);

	LayerDescription layer0;

	Group layer0Group0;
	Group layer0GlobalGroup;
	Group layer0SquareGroup;

	// first we add the mixed global group
	AddSingleNodesGroup(layer0, globalGroup, layer0GlobalGroup, 0.2f);

	// mixed square group
	AddSingleNodesGroup(layer0, squareGroup, layer0SquareGroup, 0.2f);

	// pass through group 0 (this contains game phase and move-specific information)
	AddSingleNodesGroup(layer0, group0, layer0Group0, 0.5f);

	//layerSizes.push_back(layer0.layerSize);
	//connMatrices.push_back(layer0.connections);

	layerSizes.push_back(256);
	connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

	// in the second layer, we just fully connect everything
	layerSizes.push_back(64);
	connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

	// fully connected output layer
	connMatrices.push_back(std::vector<Eigen::Triplet<float> >());

	return MoveEvalNet(inputDims, outputDims, layerSizes, connMatrices);
}

}

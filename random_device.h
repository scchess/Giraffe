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

#ifndef RANDOM_DEVICE_H
#define RANDOM_DEVICE_H

#include <random>
#include "mutex_wrapper.h"
#include "thread_wrapper.h"

#include <cstdint>

// thread-safe wrapper for std::random_device, with convenience functions
class RandomDevice
{
public:
	std::random_device::result_type operator()()
	{
		std::lock_guard<std::mutex> l(m_mutex);

		return m_rd();
	}

	std::mt19937 MakeMT()
	{
		std::lock_guard<std::mutex> l(m_mutex);

		return std::mt19937(m_rd());
	}

private:
	std::mutex m_mutex;
	std::random_device m_rd;
};

extern RandomDevice gRd;

#endif // RANDOM_DEVICE_H

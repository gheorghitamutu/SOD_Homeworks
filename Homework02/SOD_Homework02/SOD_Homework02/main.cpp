#include "Generator.h"
#include "Solver.h"

#include <thread> // hardware_concurrency

int main()
{
	std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> CASES_LIST =
	{
		{10, 10, 10},         // 10x10
		{10, 10, 100},		  // 10x100
		{10, 10, 1000},		  // 10x1000
		{10, 100, 10},		  // 10x10
		{10, 100, 100},		  // 10x100
		{10, 100, 1000},	  // 10x1000
		{10, 1000, 10},		  // 10x10
		{10, 1000, 100},	  // 10x100
		{10, 1000, 1000},	  // 10x1000
							  // 
		{100, 10, 10},		  // 100x10
		{100, 10, 100},		  // 100x100
		{100, 10, 1000},	  // 100x1000
		{100, 100, 10},		  // 100x10
		{100, 100, 100},	  // 100x100
		{100, 100, 1000},	  // 100x1000
		{100, 1000, 10},	  // 100x10
		{100, 1000, 100},	  // 100x100
		// {100, 1000, 1000}, // 100x1000
							  // 
		{1000, 10, 10},		  // 1000x10
		{1000, 10, 100},	  // 1000x100
		{1000, 10, 1000},	  // 1000x1000
		{1000, 100, 10},	  // 1000x10
		{1000, 100, 100},	  // 1000x100
		//{1000, 100, 1000},  // 1000x1000
		//{1000, 1000, 10},	  // 1000x10
		//{1000, 1000, 100},  // 1000x100
		//{1000, 1000, 1000}  // 1000x1000
	};

	constexpr auto MAX_SERIES = 10;
	const auto MAX_THREADS = std::thread::hardware_concurrency();

	constexpr auto flags =
		static_cast<uint32_t>(Solver::Solution::V00)
		| static_cast<uint32_t>(Solver::Solution::V01)
		| static_cast<uint32_t>(Solver::Solution::V02)
		| static_cast<uint32_t>(Solver::Solution::V03)
		| static_cast<uint32_t>(Solver::Solution::V04)
		| static_cast<uint32_t>(Solver::Solution::V05)
		| static_cast<uint32_t>(Solver::Solution::V06)
		| static_cast<uint32_t>(Solver::Solution::V07)
		| static_cast<uint32_t>(Solver::Solution::V08)
		| static_cast<uint32_t>(Solver::Solution::V09)
		| static_cast<uint32_t>(Solver::Solution::V10)
		| static_cast<uint32_t>(Solver::Solution::V11)
		| static_cast<uint32_t>(Solver::Solution::V12)
		| static_cast<uint32_t>(Solver::Solution::V13)
		| static_cast<uint32_t>(Solver::Solution::V14)
		| static_cast<uint32_t>(Solver::Solution::V15)
		| static_cast<uint32_t>(Solver::Solution::V16)
		| static_cast<uint32_t>(Solver::Solution::V17);

	auto i = 0u;
	for (const auto& [n, m, r] : CASES_LIST)
	{
		std::cout << "CASE #" << i++ << std::endl;

		Generator g;
		auto matrixes = g.GetMatrixes({ n, m , r });

		for (auto t = 1U; t <= MAX_THREADS; t++)
		{
			std::cout << "SERIES #" << MAX_SERIES << " THREADS #" << t << std::endl;

			Solver solver(flags, MAX_SERIES, t);
			solver.Solve(matrixes);
		}
	}

	return 0;
}

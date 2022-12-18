#pragma once

#include <omp.h>

#include <tuple>
#include <vector>
#include <initializer_list>
#include <chrono>
#include <numeric>
#include <iostream>


class Solver
{
public:
	enum Solution : uint32_t // solution flags
	{
		V00 = 0x00000000, // Sequential
		V01 = 0x00000001, // 1st for
		V02 = 0x00000002, // 2nd for
		V03 = 0x00000004, // 3rd for
		V04 = 0x00000008, // schedule(runtime) collapse(3)
		V05 = 0x00000010, // schedule(runtime) collapse(2) first 2
		V06 = 0x00000020, // schedule(runtime) collapse(2) last 2
		V07 = 0x00000040, // schedule(runtime) for 1st parrallel 2nd
		V08 = 0x00000080, // schedule(runtime) for 1st parrallel 3rd
		V09 = 0x00000100, // schedule(runtime) for 2nd parrallel 3rd
		V10 = 0x00000200, // schedule(runtime) for 2nd parrallel 3rd
		V11 = 0x00000400, // schedule(runtime) collapse(2) first 2 parallel last
		V12 = 0x00000800, // schedule(runtime) parallel 1st collapse(2) last 2
		V13 = 0x00001000, // reduction V #03
		V14 = 0x00002000, // reduction V #08
		V15 = 0x00004000, // reduction V #09
		V16 = 0x00008000, // reduction V #10
		V17 = 0x00010000, // reduction V #11
	};

private:
	// secvential
	void SolveV00(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes);

	// 1st for
	void SolveV01(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes);

	// 2nd for
	void SolveV02(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes);

	// 3rd for
	void SolveV03(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes);

	// schedule(runtime) collapse(3)
	void SolveV04(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes);

	// schedule(runtime) collapse(2) first 2
	void SolveV05(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes);

	// schedule(runtime) collapse(2) last 2
	void SolveV06(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes);

	// schedule(runtime) for 1st parrallel 2nd
	void SolveV07(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes);

	// schedule(runtime) for 1st parrallel 3rd
	void SolveV08(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes);

	// schedule(runtime) for 2nd parrallel 3rd
	void SolveV09(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes);

	// schedule(runtime) for 2nd parrallel 3rd
	void SolveV10(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes);

	// schedule(runtime) collapse(2) first 2 parallel last
	void SolveV11(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes);

	// schedule(runtime) parallel 1st collapse(2) last 2
	void SolveV12(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes);

	// reduction V #03
	void SolveV13(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes);

	// reduction V #08
	void SolveV14(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes);

	// reduction V #09
	void SolveV15(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes);

	// reduction V #10
	void SolveV16(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes);

	// reduction V #11
	void SolveV17(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes);

private:
	const uint32_t flags;
	const uint32_t maxRun;
	const uint32_t threadsCount;

	inline static const auto FLAG_LIST =
		std::initializer_list<uint32_t>
	{
			Solution::V00,
			Solution::V01,
			Solution::V02,
			Solution::V03,
			Solution::V04,
			Solution::V05,
			Solution::V06,
			Solution::V07,
			Solution::V08,
			Solution::V09,
			Solution::V10,
			Solution::V11,
			Solution::V12,
			Solution::V13,
			Solution::V14,
			Solution::V15,
			Solution::V16,
			Solution::V17,
	};


	struct CaseDuration
	{
		std::vector<uint64_t> runs;
		uint64_t best{ 0 };
		uint64_t average{ 0 };
		uint64_t worst{ 0 };

		void ComputeTimes(uint32_t seriesCount)
		{
			best = *std::min_element(runs.begin(), runs.begin() + seriesCount);
			worst = *std::max_element(runs.begin(), runs.begin() + seriesCount);
			average = std::reduce(runs.begin(), runs.end()) / seriesCount;
		}
	};
	std::vector<CaseDuration> cases; // tids => b/a/w + times (split in tids)

	uint32_t currentRun{ 0U };

public:
	Solver(uint32_t flags, uint32_t maxRun, uint32_t threadsCount);
	bool Solve(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>>& matrixes);
};

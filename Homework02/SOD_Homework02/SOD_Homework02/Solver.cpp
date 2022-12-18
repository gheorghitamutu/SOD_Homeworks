#include "Solver.h"

static std::vector<std::vector<uint32_t>> resultMatrix;
static void SanityCheck(const std::vector<std::vector<uint32_t>>& res, uint32_t version)
{
	if (res != resultMatrix)
	{
		std::cout << "Failed V" << version << "!" << std::endl;
	}
}

void Solver::SolveV00(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes)
{
	const auto start = std::chrono::high_resolution_clock::now();

	auto& [a, b, c] = matrixes;

	const auto n = a.size(), m = b.size(), r = c.at(0).size();

	auto& currentCase = cases.at(0);

	// https://users.ece.cmu.edu/~franzf/papers/gttse07.pdf - MMM
	for (size_t i = 0U; i < n; i++)
	{
		auto& row = c.at(i);
		auto& aRow = a.at(i);
		for (size_t j = 0U; j < r; j++)
		{
			auto& cell = row.at(j);
			for (size_t k = 0; k < m; k++)
			{
				cell += aRow.at(k) * b.at(k).at(j);
			}
		}
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	currentCase.runs.at(static_cast<size_t>(currentRun)) = static_cast<uint64_t>(duration.count());

	resultMatrix = std::move(c); // use it for checks
}

void Solver::SolveV01(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes)
{
	const auto start = std::chrono::high_resolution_clock::now();

	const auto& a = std::get<0>(matrixes);
	const auto& b = std::get<1>(matrixes);
	auto& c = std::get<2>(matrixes);

	const auto n = a.size(), m = b.size(), r = c.at(0).size();
	size_t i, j, k;
	auto& currentCase = cases.at(1);

#pragma omp parallel shared(c) private(i, j, k)
	{
#pragma omp for nowait
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < r; j++)
			{
				for (k = 0; k < m; k++)
				{
					c.at(i).at(j) += a.at(i).at(k) * b.at(k).at(j);
				}
			}
		}
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	currentCase.runs.at(static_cast<size_t>(currentRun)) = static_cast<uint64_t>(duration.count());

	SanityCheck(c, 1);
}

void Solver::SolveV02(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes)
{
	const auto start = std::chrono::high_resolution_clock::now();

	const auto& a = std::get<0>(matrixes);
	const auto& b = std::get<1>(matrixes);
	auto& c = std::get<2>(matrixes);

	const auto n = a.size(), m = b.size(), r = c.at(0).size();
	size_t i, j, k;
	auto& currentCase = cases.at(2);

#pragma omp parallel shared(c) private(i, j, k)
	{
		for (i = 0; i < n; i++)
		{
#pragma omp for nowait
			for (j = 0; j < r; j++)
			{
				for (k = 0; k < m; k++)
				{
					c.at(i).at(j) += a.at(i).at(k) * b.at(k).at(j);
				}
			}
		}
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	currentCase.runs.at(static_cast<size_t>(currentRun)) = static_cast<uint64_t>(duration.count());

	SanityCheck(c, 2);
}

void Solver::SolveV03(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes)
{
	const auto start = std::chrono::high_resolution_clock::now();

	const auto& a = std::get<0>(matrixes);
	const auto& b = std::get<1>(matrixes);
	auto& c = std::get<2>(matrixes);

	const auto n = a.size(), m = b.size(), r = c.at(0).size();
	size_t i, j, k;
	auto& currentCase = cases.at(3);

#pragma omp parallel shared(c) private(i, j, k)
	{
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < r; j++)
			{
#pragma omp for nowait
				for (k = 0; k < m; k++)
				{
#pragma omp critical
					c.at(i).at(j) += a.at(i).at(k) * b.at(k).at(j);
				}
			}
		}
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	currentCase.runs.at(static_cast<size_t>(currentRun)) = static_cast<uint64_t>(duration.count());

	SanityCheck(c, 3);
}

void Solver::SolveV04(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes)
{
	const auto start = std::chrono::high_resolution_clock::now();

	const auto& a = std::get<0>(matrixes);
	const auto& b = std::get<1>(matrixes);
	auto& c = std::get<2>(matrixes);

	const auto n = a.size(), m = b.size(), r = c.at(0).size();
	size_t i, j, k;
	auto& currentCase = cases.at(4);

#pragma omp parallel shared(c) private(i, j, k)
	{
#pragma omp for schedule(runtime) collapse(3) nowait
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < r; j++)
			{
				for (k = 0; k < m; k++)
				{
#pragma omp critical
					c.at(i).at(j) += a.at(i).at(k) * b.at(k).at(j);
				}
			}
		}
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	currentCase.runs.at(static_cast<size_t>(currentRun)) = static_cast<uint64_t>(duration.count());

	SanityCheck(c, 4);
}

void Solver::SolveV05(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes)
{
	const auto start = std::chrono::high_resolution_clock::now();

	const auto& a = std::get<0>(matrixes);
	const auto& b = std::get<1>(matrixes);
	auto& c = std::get<2>(matrixes);

	const auto n = a.size(), m = b.size(), r = c.at(0).size();
	size_t i, j, k;
	auto& currentCase = cases.at(5);

#pragma omp parallel shared(c) private(i, j, k)
	{
#pragma omp for schedule(runtime) collapse(2) nowait
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < r; j++)
			{
				for (k = 0; k < m; k++)
				{
					c.at(i).at(j) += a.at(i).at(k) * b.at(k).at(j);
				}
			}
		}
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	currentCase.runs.at(static_cast<size_t>(currentRun)) = static_cast<uint64_t>(duration.count());

	SanityCheck(c, 5);
}

void Solver::SolveV06(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes)
{
	const auto start = std::chrono::high_resolution_clock::now();

	const auto& a = std::get<0>(matrixes);
	const auto& b = std::get<1>(matrixes);
	auto& c = std::get<2>(matrixes);

	const auto n = a.size(), m = b.size(), r = c.at(0).size();
	size_t i, j, k;
	auto& currentCase = cases.at(6);

#pragma omp parallel shared(c) private(i, j, k)
	{
		for (i = 0; i < n; i++)
		{
#pragma omp for schedule(runtime) collapse(2) nowait
			for (j = 0; j < r; j++)
			{
				for (k = 0; k < m; k++)
				{
#pragma omp critical
					c.at(i).at(j) += a.at(i).at(k) * b.at(k).at(j);
				}
			}
		}
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	currentCase.runs.at(static_cast<size_t>(currentRun)) = static_cast<uint64_t>(duration.count());

	SanityCheck(c, 6);
}

void Solver::SolveV07(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes)
{
	const auto start = std::chrono::high_resolution_clock::now();

	const auto& a = std::get<0>(matrixes);
	const auto& b = std::get<1>(matrixes);
	auto& c = std::get<2>(matrixes);

	const auto n = a.size(), m = b.size(), r = c.at(0).size();
	size_t i, j, k;
	auto& currentCase = cases.at(7);

#pragma omp parallel shared(c) private(i, j, k)
	{
#pragma omp for nowait
		for (i = 0; i < n; i++)
		{
#pragma omp parallel for schedule(runtime) private(j, k) 
			for (j = 0; j < r; j++)
			{
				for (k = 0; k < m; k++)
				{
					c.at(i).at(j) += a.at(i).at(k) * b.at(k).at(j);
				}
			}
		}
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	currentCase.runs.at(static_cast<size_t>(currentRun)) = static_cast<uint64_t>(duration.count());

	SanityCheck(c, 7);
}

void Solver::SolveV08(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes)
{
	const auto start = std::chrono::high_resolution_clock::now();

	const auto& a = std::get<0>(matrixes);
	const auto& b = std::get<1>(matrixes);
	auto& c = std::get<2>(matrixes);

	const auto n = a.size(), m = b.size(), r = c.at(0).size();
	size_t i, j, k;
	auto& currentCase = cases.at(8);

#pragma omp parallel shared(c) private(i, j, k)
	{
#pragma omp for nowait
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < r; j++)
			{
#pragma omp parallel for schedule(runtime) private(k)
				for (k = 0; k < m; k++)
				{
#pragma omp critical
					c.at(i).at(j) += a.at(i).at(k) * b.at(k).at(j);
				}
			}
		}
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	currentCase.runs.at(static_cast<size_t>(currentRun)) = static_cast<uint64_t>(duration.count());

	SanityCheck(c, 8);
}

void Solver::SolveV09(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes)
{
	const auto start = std::chrono::high_resolution_clock::now();

	const auto& a = std::get<0>(matrixes);
	const auto& b = std::get<1>(matrixes);
	auto& c = std::get<2>(matrixes);

	const auto n = a.size(), m = b.size(), r = c.at(0).size();
	size_t i, j, k;
	auto& currentCase = cases.at(9);

#pragma omp parallel shared(c) private(i, j, k)
	{
		for (i = 0; i < n; i++)
		{
#pragma omp for nowait
			for (j = 0; j < r; j++)
			{
#pragma omp parallel for schedule(runtime) private(k)
				for (k = 0; k < m; k++)
				{
#pragma omp critical
					c.at(i).at(j) += a.at(i).at(k) * b.at(k).at(j);
				}
			}
		}
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	currentCase.runs.at(static_cast<size_t>(currentRun)) = static_cast<uint64_t>(duration.count());

	SanityCheck(c, 9);
}

void Solver::SolveV10(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes)
{
	const auto start = std::chrono::high_resolution_clock::now();

	const auto& a = std::get<0>(matrixes);
	const auto& b = std::get<1>(matrixes);
	auto& c = std::get<2>(matrixes);

	const auto n = a.size(), m = b.size(), r = c.at(0).size();
	size_t i, j, k;
	auto& currentCase = cases.at(10);

#pragma omp parallel shared(c) private(i, j, k)
	{
#pragma omp for nowait
		for (i = 0; i < n; i++)
		{
#pragma omp parallel for schedule(runtime) private(j, k)
			for (j = 0; j < r; j++)
			{
#pragma omp parallel for schedule(runtime) private(k)
				for (k = 0; k < m; k++)
				{
#pragma omp critical
					c.at(i).at(j) += a.at(i).at(k) * b.at(k).at(j);
				}
			}
		}
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	currentCase.runs.at(static_cast<size_t>(currentRun)) = static_cast<uint64_t>(duration.count());

	SanityCheck(c, 10);
}

void Solver::SolveV11(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes)
{
	const auto start = std::chrono::high_resolution_clock::now();

	const auto& a = std::get<0>(matrixes);
	const auto& b = std::get<1>(matrixes);
	auto& c = std::get<2>(matrixes);

	const auto n = a.size(), m = b.size(), r = c.at(0).size();
	size_t i, j, k;
	auto& currentCase = cases.at(11);

#pragma omp parallel shared(c) private(i, j, k)
	{
#pragma omp for schedule(runtime) collapse(2) nowait
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < r; j++)
			{
#pragma omp parallel for
				for (k = 0; k < m; k++)
				{
#pragma omp critical
					c.at(i).at(j) += a.at(i).at(k) * b.at(k).at(j);
				}
			}
		}
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	currentCase.runs.at(static_cast<size_t>(currentRun)) = static_cast<uint64_t>(duration.count());

	SanityCheck(c, 11);
}

void Solver::SolveV12(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes)
{
	const auto start = std::chrono::high_resolution_clock::now();

	const auto& a = std::get<0>(matrixes);
	const auto& b = std::get<1>(matrixes);
	auto& c = std::get<2>(matrixes);

	const auto n = a.size(), m = b.size(), r = c.at(0).size();
	size_t i, j, k;
	auto& currentCase = cases.at(12);

#pragma omp parallel shared(c) private(i, j, k)
	{
#pragma omp for nowait
		for (i = 0; i < n; i++)
		{
#pragma omp parallel for schedule(runtime) collapse(2)
			for (j = 0; j < r; j++)
			{
				for (k = 0; k < m; k++)
				{
#pragma omp critical
					c.at(i).at(j) += a.at(i).at(k) * b.at(k).at(j);
				}
			}
		}
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	currentCase.runs.at(static_cast<size_t>(currentRun)) = static_cast<uint64_t>(duration.count());

	SanityCheck(c, 12);
}

void Solver::SolveV13(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes)
{
	const auto start = std::chrono::high_resolution_clock::now();

	const auto& a = std::get<0>(matrixes);
	const auto& b = std::get<1>(matrixes);
	auto& c = std::get<2>(matrixes);

	const auto n = a.size(), m = b.size(), r = c.at(0).size();
	size_t i, j, k;
	auto& currentCase = cases.at(13);

#pragma omp parallel shared(c) private(i, j, k)
	{
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < r; j++)
			{
				uint32_t sum = 0;
#pragma omp parallel for reduction(+: sum) private(k)
				for (k = 0; k < m; k++)
				{
					sum += a.at(i).at(k) * b.at(k).at(j);
				}
				c.at(i).at(j) = sum;
			}
		}
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	currentCase.runs.at(static_cast<size_t>(currentRun)) = static_cast<uint64_t>(duration.count());

	SanityCheck(c, 13);
}

void Solver::SolveV14(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes)
{
	const auto start = std::chrono::high_resolution_clock::now();

	const auto& a = std::get<0>(matrixes);
	const auto& b = std::get<1>(matrixes);
	auto& c = std::get<2>(matrixes);

	const auto n = a.size(), m = b.size(), r = c.at(0).size();
	size_t i, j, k;
	auto& currentCase = cases.at(14);

#pragma omp parallel shared(c) private(i, j, k)
	{
#pragma omp for nowait
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < r; j++)
			{
				uint32_t sum = 0;
#pragma omp parallel for schedule(runtime) reduction(+: sum) private(k)
				for (k = 0; k < m; k++)
				{
					sum += a.at(i).at(k) * b.at(k).at(j);
				}
				c.at(i).at(j) = sum;
			}
		}
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	currentCase.runs.at(static_cast<size_t>(currentRun)) = static_cast<uint64_t>(duration.count());

	SanityCheck(c, 14);
}

void Solver::SolveV15(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes)
{
	const auto start = std::chrono::high_resolution_clock::now();

	const auto& a = std::get<0>(matrixes);
	const auto& b = std::get<1>(matrixes);
	auto& c = std::get<2>(matrixes);

	const auto n = a.size(), m = b.size(), r = c.at(0).size();
	size_t i, j, k;
	auto& currentCase = cases.at(15);

#pragma omp parallel shared(c) private(i, j, k)
	{
		for (i = 0; i < n; i++)
		{
#pragma omp for nowait
			for (j = 0; j < r; j++)
			{
				uint32_t sum = 0;
#pragma omp parallel for schedule(runtime) reduction(+: sum) private(k)
				for (k = 0; k < m; k++)
				{
					sum += a.at(i).at(k) * b.at(k).at(j);
				}
				c.at(i).at(j) = sum;
			}
		}
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	currentCase.runs.at(static_cast<size_t>(currentRun)) = static_cast<uint64_t>(duration.count());

	SanityCheck(c, 15);
}

void Solver::SolveV16(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes)
{
	const auto start = std::chrono::high_resolution_clock::now();

	const auto& a = std::get<0>(matrixes);
	const auto& b = std::get<1>(matrixes);
	auto& c = std::get<2>(matrixes);

	const auto n = a.size(), m = b.size(), r = c.at(0).size();
	size_t i, j, k;
	auto& currentCase = cases.at(16);

#pragma omp parallel shared(c) private(i, j, k)
	{
#pragma omp for nowait
		for (i = 0; i < n; i++)
		{
#pragma omp parallel for schedule(runtime) private(j, k)
			for (j = 0; j < r; j++)
			{
				uint32_t sum = 0;
#pragma omp parallel for schedule(runtime) reduction(+: sum) private(k)
				for (k = 0; k < m; k++)
				{
					sum += a.at(i).at(k) * b.at(k).at(j);
				}
				c.at(i).at(j) = sum;
			}
		}
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	currentCase.runs.at(static_cast<size_t>(currentRun)) = static_cast<uint64_t>(duration.count());

	SanityCheck(c, 16);
}

void Solver::SolveV17(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> matrixes)
{
	const auto start = std::chrono::high_resolution_clock::now();

	const auto& a = std::get<0>(matrixes);
	const auto& b = std::get<1>(matrixes);
	auto& c = std::get<2>(matrixes);

	const auto n = a.size(), m = b.size(), r = c.at(0).size();
	size_t i, j, k;
	auto& currentCase = cases.at(17);

#pragma omp parallel shared(c) private(i, j, k)
	{
#pragma omp for schedule(runtime) collapse(2) nowait
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < r; j++)
			{
				uint32_t sum = 0;
#pragma omp parallel for schedule(runtime) reduction(+: sum) private(k)
				for (k = 0; k < m; k++)
				{
					sum += a.at(i).at(k) * b.at(k).at(j);
				}
				c.at(i).at(j) = sum;
			}
		}
	}

	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	currentCase.runs.at(static_cast<size_t>(currentRun)) = static_cast<uint64_t>(duration.count());

	SanityCheck(c, 17);
}

Solver::Solver(uint32_t _flags, uint32_t series, uint32_t threadsNo) : flags(_flags), maxRun(series), threadsCount(threadsNo)
{
	cases.insert(cases.end(), FLAG_LIST.size(), {});
	for (auto& c : cases)
	{
		c.runs.insert(c.runs.end(), maxRun, 0);
	}

	omp_set_num_threads(static_cast<int32_t>(threadsCount));

	const auto levels = omp_get_max_active_levels();
	omp_set_max_active_levels(levels);
}

bool Solver::Solve(std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>>& matrixes)
{
	for (currentRun = 0U; currentRun < maxRun; currentRun++)
	{
		std::cout << "Processing RUN #" << currentRun << std::endl;
		for (const auto flag : FLAG_LIST)
		{
			if ((flags & flag) == flag)
			{
				switch (static_cast<Solution>(flag))
				{
				case Solution::V00: SolveV00(matrixes); break;
				case Solution::V01: SolveV01(matrixes); break;
				case Solution::V02: SolveV02(matrixes); break;
				case Solution::V03: SolveV03(matrixes); break;
				case Solution::V04: SolveV04(matrixes); break;
				case Solution::V05: SolveV05(matrixes); break;
				case Solution::V06: SolveV06(matrixes); break;
				case Solution::V07: SolveV07(matrixes); break;
				case Solution::V08: SolveV08(matrixes); break;
				case Solution::V09: SolveV09(matrixes); break;
				case Solution::V10: SolveV10(matrixes); break;
				case Solution::V11: SolveV11(matrixes); break;
				case Solution::V12: SolveV12(matrixes); break;
				case Solution::V13: SolveV13(matrixes); break;
				case Solution::V14: SolveV14(matrixes); break;
				case Solution::V15: SolveV15(matrixes); break;
				case Solution::V16: SolveV16(matrixes); break;
				case Solution::V17: SolveV17(matrixes); break;
				}
			}
		}
	}

	for (auto fi = 0U; fi < FLAG_LIST.size(); fi++)
	{
		const auto& flag = *(FLAG_LIST.begin() + fi);

		if ((flags & flag) == flag)
		{
			auto& c = cases.at(fi);
			c.ComputeTimes(maxRun);

			std::cout << "V #" << fi
				<< " Best: " << c.best
				<< " Average: " << c.average
				<< " Worst: " << c.worst
				<< std::endl;

			std::cout << "Runs: [";
			for (const auto& t : c.runs)
			{
				std::cout << t << ", ";
			}
			std::cout << "]" << std::endl;
		}
	}

	std::cout << std::endl;

	return true;
}

#include "Generator.h"

std::seed_seq Generator::GetRandomSeed()
{
	std::random_device source;
	uint32_t random_data[SIZE_RANDOM_ARRAY]{ 0 };
	for (auto& elem : random_data)
	{
		elem = source();
	}

	return std::seed_seq(random_data + 0, random_data + SIZE_RANDOM_ARRAY);
}

uint32_t Generator::GetRandomNumber(std::uniform_int_distribution<uint32_t>& dist)
{
	return dist(rng);
}

const std::tuple<uint32_t, uint32_t, uint32_t> Generator::GetDimensions()
{
	std::uniform_int_distribution<uint32_t> dist(DIMENSION_LOWER_LIMIT, DIMENSION_HIGHER_LIMIT);
	return { dist(rng), dist(rng), dist(rng) };
}

void Generator::PopulateMatrix(const std::pair<uint32_t, uint32_t>& dimensions, std::vector<std::vector<uint32_t>>& matrix)
{
	std::uniform_int_distribution<uint32_t> dist(MIN_VALUE_MATRIX_VALUE, MAX_VALUE_MATRIX_VALUE);

	const auto& [x, y] = dimensions;

	matrix.reserve(x);
	for (auto i = 0U; i < x; i++) {
		auto& row = matrix.emplace_back();
		row.reserve(y);

		for (auto j = 0U; j < y; j++) {
			row.emplace_back(dist(rng));
		}
	}
}

std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> Generator::GetMatrixes()
{
	const auto& [n, m, r] = GetDimensions();
	return GenerateMatrixes(n, m, r);
}


std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> Generator::GetMatrixes(const std::tuple<uint32_t, uint32_t, uint32_t>& dimensions)
{
	const auto& [n, m, r] = dimensions;
	return GenerateMatrixes(n, m, r);
}

std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> Generator::GenerateMatrixes(uint32_t n, uint32_t m, uint32_t r)
{
	std::vector <std::vector<uint32_t>> a;
	PopulateMatrix({ n, m }, a);

	std::vector <std::vector<uint32_t>> b;
	PopulateMatrix({ m, r }, b);

	std::vector <std::vector<uint32_t>> c;
	c.resize(n);
	for (auto& e : c)
	{
		e.insert(e.end(), r, 0);
	}


	std::cout << "Generated matrix A[" << n << "][" << m << "], B[" << m << "][" << r << "] and C[" << n << "][" << r << "]." << std::endl << std::endl;

	return { a, b, c };
}

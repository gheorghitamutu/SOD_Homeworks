#pragma once

#include <random>
#include <tuple>
#include <iostream>

constexpr auto DIMENSION_LOWER_LIMIT = 100U;
constexpr auto DIMENSION_HIGHER_LIMIT = 10000U;
constexpr auto SIZE_RANDOM_ARRAY = 10U;
constexpr auto MIN_VALUE_MATRIX_VALUE = 0U;
constexpr auto MAX_VALUE_MATRIX_VALUE = 1000U;

class Generator
{
private:
	uint32_t GetRandomNumber(std::uniform_int_distribution<uint32_t>& dist);
	const std::tuple<uint32_t, uint32_t, uint32_t> GetDimensions();

	std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> GenerateMatrixes(uint32_t n, uint32_t m, uint32_t r);
	void PopulateMatrix(const std::pair<uint32_t, uint32_t>& dimensions, std::vector<std::vector<uint32_t>>& matrix);

private:
	static std::seed_seq GetRandomSeed();

private:
	inline static std::seed_seq seed{ GetRandomSeed() };
	inline static std::default_random_engine rng{ seed };

public:
	std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> GetMatrixes();
	std::tuple<const std::vector<std::vector<uint32_t>>, const std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> GetMatrixes(const std::tuple<uint32_t, uint32_t, uint32_t>& dimensions);
};

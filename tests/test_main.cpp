#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "imagenn/version.hpp"

/// @file test_main.cpp
/// @brief Unit tests. This skeleton test only verifies the build/test pipeline;
/// real coverage is added together with each feature.

TEST_CASE("project version is non-empty") {
    CHECK(imagenn::project_version() == "1.0.0");
    CHECK_FALSE(imagenn::project_version().empty());
}

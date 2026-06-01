#include <exception>
#include <iostream>

#include "imagenn/version.hpp"

/// @file main.cpp
/// @brief Command-line entry point.
///
/// At this skeleton stage the executable only proves that the build pipeline
/// (library + executable + tests) works end to end. The full command-line
/// interface is added in later stages.

int main(int argc, char** argv) {
    try {
        std::cout << "ImageNN C++ " << imagenn::project_version() << "\n";
        if (argc > 1) {
            std::cout << "Arguments are not handled yet (skeleton stage).\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

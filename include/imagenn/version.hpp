#ifndef IMAGENN_VERSION_HPP
#define IMAGENN_VERSION_HPP

#include <string>

/// @file version.hpp
/// @brief Project version information.

/// @namespace imagenn
/// @brief Root namespace for all components of the ImageNN C++ port.
namespace imagenn {

/// @brief Returns the human-readable project version string.
/// @return Semantic version of the project, e.g. "1.0.0".
std::string project_version();

} // namespace imagenn

#endif // IMAGENN_VERSION_HPP

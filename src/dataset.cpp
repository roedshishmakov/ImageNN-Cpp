#include "imagenn/dataset.hpp"

#include <cctype>
#include <filesystem>
#include <string>

#include "imagenn/exceptions.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

namespace imagenn {

namespace {

bool is_image_file(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    for (std::size_t i = 0; i < ext.size(); ++i) {
        ext[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(ext[i])));
    }
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg";
}

std::vector<std::filesystem::path> list_images(const std::string& directory) {
    if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) {
        throw PathError("Directory does not exist: " + directory);
    }

    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && is_image_file(entry.path())) {
            files.push_back(entry.path());
        }
    }
    // Простая сортировка вставками по имени файла.
    for (std::size_t i = 1; i < files.size(); ++i) {
        std::filesystem::path current = files[i];
        std::size_t j = i;
        while (j > 0 && files[j - 1] > current) {
            files[j] = files[j - 1];
            --j;
        }
        files[j] = current;
    }
    return files;
}

int label_from_name(const std::filesystem::path& path) {
    const std::string name = path.filename().string();
    const std::size_t underscore = name.find('_');
    try {
        return std::stoi(name.substr(0, underscore));
    } catch (const std::exception&) {
        throw ValidationError("Cannot parse class label from file name: " + name);
    }
}

} // namespace

std::vector<double> image_to_input(const std::string& path) {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (pixels == nullptr) {
        throw PathError("Cannot read image: " + path);
    }

    std::vector<unsigned char> resized(static_cast<std::size_t>(kImageSize) * kImageSize * 3);
    stbir_resize_uint8_linear(pixels, width, height, 0, resized.data(), kImageSize, kImageSize, 0,
                              STBIR_RGB);
    stbi_image_free(pixels);

    std::vector<double> result;
    result.reserve(static_cast<std::size_t>(kInputSize));
    for (int i = 0; i < kInputSize; ++i) {
        const std::size_t base = static_cast<std::size_t>(i) * 3;
        const double mean = (resized[base] + resized[base + 1] + resized[base + 2]) / 3.0;
        result.push_back(mean < kPixelThreshold ? 1.0 : 0.0);
    }
    return result;
}

std::vector<NamedInput> load_inputs(const std::string& directory) {
    std::vector<NamedInput> dataset;
    for (const auto& path : list_images(directory)) {
        dataset.push_back({path.filename().string(), image_to_input(path.string())});
    }
    return dataset;
}

std::vector<TrainingExample> load_training_examples(const std::string& directory) {
    std::vector<TrainingExample> dataset;
    for (const auto& path : list_images(directory)) {
        const int label = label_from_name(path);
        if (label < 0 || label >= kNumClasses) {
            throw ValidationError("Class label out of range in file: " + path.filename().string());
        }
        std::vector<double> target(static_cast<std::size_t>(kNumClasses), 0.0);
        target[static_cast<std::size_t>(label)] = 1.0;
        dataset.emplace_back(image_to_input(path.string()), std::move(target));
    }
    return dataset;
}

} // namespace imagenn

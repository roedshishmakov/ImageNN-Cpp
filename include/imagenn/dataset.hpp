#ifndef IMAGENN_DATASET_HPP
#define IMAGENN_DATASET_HPP

#include <string>
#include <utility>
#include <vector>

#include "imagenn/network.hpp"

/// @file dataset.hpp
/// @brief Загрузка изображений и подготовка обучающих данных.

namespace imagenn {

/// Сторона, к которой приводится изображение перед подачей в сеть.
constexpr int kImageSize = 16;
/// Размер входного вектора сети (число пикселей квадратного изображения).
constexpr int kInputSize = kImageSize * kImageSize;
/// Число распознаваемых классов (цифры 0–9).
constexpr int kNumClasses = 10;
/// Порог яркости: пиксель темнее порога считается закрашенным.
constexpr double kPixelThreshold = 170.0;

/// @brief Один загруженный для распознавания образец.
struct NamedInput {
    std::string name;           ///< Имя файла образца.
    std::vector<double> values; ///< Входной вектор длины kInputSize.
};

/// @brief Преобразует файл изображения во входной вектор сети.
/// @param path путь к файлу изображения
/// @return бинаризованный вектор длины kInputSize
/// @throws PathError если изображение не удаётся прочитать
std::vector<double> image_to_input(const std::string& path);

/// @brief Загружает изображения из папки для распознавания.
/// @param directory путь к папке с изображениями
/// @return образцы с именами файлов, отсортированные по имени
/// @throws PathError если папка не существует
std::vector<NamedInput> load_inputs(const std::string& directory);

/// @brief Загружает обучающие примеры из папки.
///
/// Метка класса берётся из имени файла до первого символа '_'.
/// @param directory путь к папке с изображениями
/// @return обучающие примеры с one-hot эталонами
/// @throws PathError если папка не существует
/// @throws ValidationError если метку в имени файла не удаётся разобрать
std::vector<TrainingExample> load_training_examples(const std::string& directory);

} // namespace imagenn

#endif // IMAGENN_DATASET_HPP

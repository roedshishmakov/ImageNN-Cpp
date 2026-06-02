#ifndef IMAGENN_CONFIG_HPP
#define IMAGENN_CONFIG_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "imagenn/network.hpp"

/// @file config.hpp
/// @brief Конфигурация архитектуры сети и параметров обучения.

namespace imagenn {

/// @brief Описание одного слоя сети.
struct LayerConfig {
    std::string type;           ///< Тип слоя (поддерживается только "dense").
    std::size_t size = 0;       ///< Количество нейронов.
    std::string activation;     ///< Имя функции активации.
    bool use_bias = false;      ///< Использовать ли нейрон смещения.
    double random_radius = 0.1; ///< Радиус инициализации весов.
};

/// @brief Параметры обучения.
struct TrainingConfig {
    double learning_rate = 0.1;    ///< Скорость обучения.
    int epochs = 10;               ///< Число эпох.
    double clip_value = 5.0;       ///< Порог ограничения дельты веса.
    bool use_cross_entropy = true; ///< Использовать ли кросс-энтропию.
};

/// @brief Полная конфигурация сети.
struct NetworkConfig {
    std::vector<LayerConfig> layers; ///< Слои сети.
    TrainingConfig training;         ///< Параметры обучения.
};

/// @brief Загружает конфигурацию из файла.
/// @param path путь к файлу конфигурации
/// @return разобранная конфигурация
/// @throws PathError если файл не найден
/// @throws ValidationError при ошибке формата
NetworkConfig parse_config_file(const std::string& path);

/// @brief Сохраняет конфигурацию в файл.
/// @param config конфигурация для сохранения
/// @param path путь к файлу
/// @throws PathError если файл не удаётся открыть на запись
void save_config(const NetworkConfig& config, const std::string& path);

/// @brief Создаёт файл-шаблон конфигурации со стандартной архитектурой.
/// @param path путь к создаваемому файлу
/// @throws PathError если файл не удаётся открыть на запись
void create_config_template(const std::string& path);

/// @brief Возвращает конфигурацию стандартной архитектуры.
/// @return конфигурация по умолчанию
NetworkConfig default_config();

/// @brief Строит сеть по конфигурации.
/// @param config конфигурация сети
/// @return построенная сеть
/// @throws ValidationError если указана неизвестная функция активации
NeuralNetwork build_network(const NetworkConfig& config);

} // namespace imagenn

#endif // IMAGENN_CONFIG_HPP

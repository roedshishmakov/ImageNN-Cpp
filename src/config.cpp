#include "imagenn/config.hpp"

#include <fstream>
#include <sstream>

#include "imagenn/activations.hpp"
#include "imagenn/dataset.hpp"
#include "imagenn/exceptions.hpp"

namespace imagenn {

namespace {

std::string trim(const std::string& text) {
    const std::size_t begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const std::size_t end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, delimiter)) {
        parts.push_back(item);
    }
    return parts;
}

bool to_bool(const std::string& value) {
    return trim(value) == "true";
}

const ActivationBase& activation_by_name(const std::string& name) {
    if (name == "relu") {
        return relu_activation();
    }
    if (name == "sigmoid") {
        return sigmoid_activation();
    }
    if (name == "softmax") {
        return softmax_activation();
    }
    if (name == "transparent") {
        return transparent_activation();
    }
    throw ValidationError("Unknown activation function: " + name);
}

LayerConfig parse_layer_line(const std::string& line, std::size_t line_number) {
    const std::vector<std::string> parts = split(line, ':');
    if (parts.size() != 5) {
        throw ValidationError("Invalid layer format at line " + std::to_string(line_number));
    }

    LayerConfig layer;
    layer.type = trim(parts[0]);
    if (layer.type != "dense") {
        throw ValidationError("Unknown layer type at line " + std::to_string(line_number));
    }
    try {
        layer.size = static_cast<std::size_t>(std::stoul(trim(parts[1])));
        layer.activation = trim(parts[2]);
        layer.use_bias = to_bool(parts[3]);
        layer.random_radius = std::stod(trim(parts[4]));
    } catch (const std::exception&) {
        throw ValidationError("Invalid layer values at line " + std::to_string(line_number));
    }
    return layer;
}

void parse_training_line(const std::string& line, TrainingConfig& training) {
    const std::size_t separator = line.find('=');
    if (separator == std::string::npos) {
        return;
    }
    const std::string key = trim(line.substr(0, separator));
    const std::string value = trim(line.substr(separator + 1));
    try {
        if (key == "learning_rate") {
            training.learning_rate = std::stod(value);
        } else if (key == "epochs") {
            training.epochs = std::stoi(value);
        } else if (key == "clip_value") {
            training.clip_value = std::stod(value);
        } else if (key == "use_cross_entropy") {
            training.use_cross_entropy = to_bool(value);
        }
    } catch (const std::exception&) {
        throw ValidationError("Invalid training value for key: " + key);
    }
}

} // namespace

NetworkConfig parse_config_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw PathError("Config file not found: " + path);
    }

    NetworkConfig config;
    bool in_training_section = false;
    std::string raw;
    std::size_t line_number = 0;

    while (std::getline(file, raw)) {
        ++line_number;
        const std::string line = trim(raw);

        if (line.empty() || line[0] == '#') {
            if (!config.layers.empty() && !in_training_section) {
                in_training_section = true;
            }
            continue;
        }

        if (!in_training_section) {
            config.layers.push_back(parse_layer_line(line, line_number));
        } else {
            parse_training_line(line, config.training);
        }
    }

    if (config.layers.empty()) {
        throw ValidationError("Configuration contains no layers");
    }
    return config;
}

void save_config(const NetworkConfig& config, const std::string& path) {
    std::ofstream file(path);
    if (!file) {
        throw PathError("Cannot open config file for writing: " + path);
    }

    file << "# Конфигурация нейронной сети\n";
    file << "# Формат: [layer_type]:[size]:[activation]:[use_bias]:[random_radius]\n\n";
    for (const LayerConfig& layer : config.layers) {
        file << "dense:" << layer.size << ":" << layer.activation << ":"
             << (layer.use_bias ? "true" : "false") << ":" << layer.random_radius << "\n";
    }

    file << "\n# Параметры обучения\n";
    file << "learning_rate=" << config.training.learning_rate << "\n";
    file << "epochs=" << config.training.epochs << "\n";
    file << "clip_value=" << config.training.clip_value << "\n";
    file << "use_cross_entropy=" << (config.training.use_cross_entropy ? "true" : "false") << "\n";
}

void create_config_template(const std::string& path) {
    save_config(default_config(), path);
}

NetworkConfig default_config() {
    NetworkConfig config;
    config.layers.push_back({"dense", 32, "relu", true, 0.1});
    config.layers.push_back({"dense", 32, "relu", true, 0.1});
    config.layers.push_back({"dense", 10, "softmax", false, 0.1});
    return config;
}

NeuralNetwork build_network(const NetworkConfig& config) {
    NeuralNetwork network;
    network.add_input_layer(kInputSize);
    for (const LayerConfig& layer : config.layers) {
        network.add_layer(layer.size, activation_by_name(layer.activation), layer.random_radius,
                          layer.use_bias);
    }
    return network;
}

} // namespace imagenn

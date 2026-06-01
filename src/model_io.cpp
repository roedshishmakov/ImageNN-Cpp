#include "imagenn/model_io.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

#include "imagenn/exceptions.hpp"

namespace imagenn {

namespace {
constexpr char kMagic[] = "IMAGENN_MODEL";
constexpr int kFormatVersion = 1;
} // namespace

void save_model(const NeuralNetwork& network, const std::string& path) {
    std::ofstream file(path);
    if (!file) {
        throw PathError("Cannot open model file for writing: " + path);
    }

    const std::vector<std::vector<std::vector<double>>> weights = network.export_weights();
    file << std::setprecision(std::numeric_limits<double>::max_digits10);
    file << kMagic << " " << kFormatVersion << "\n";
    file << "LAYERS " << weights.size() << "\n";

    for (const auto& layer : weights) {
        file << "LAYER " << layer.size() << "\n";
        for (const auto& neuron : layer) {
            file << neuron.size();
            for (double weight : neuron) {
                file << " " << weight;
            }
            file << "\n";
        }
    }
}

void load_model(NeuralNetwork& network, const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw PathError("Model file not found: " + path);
    }

    std::string magic;
    int version = 0;
    if (!(file >> magic >> version) || magic != kMagic || version != kFormatVersion) {
        throw ValidationError("Unsupported model file format: " + path);
    }

    std::string token;
    std::size_t layer_count = 0;
    if (!(file >> token >> layer_count) || token != "LAYERS") {
        throw ValidationError("Malformed model header: " + path);
    }

    std::vector<std::vector<std::vector<double>>> weights;
    weights.reserve(layer_count);
    for (std::size_t l = 0; l < layer_count; ++l) {
        std::size_t neuron_count = 0;
        if (!(file >> token >> neuron_count) || token != "LAYER") {
            throw ValidationError("Malformed layer record: " + path);
        }
        std::vector<std::vector<double>> layer;
        layer.reserve(neuron_count);
        for (std::size_t n = 0; n < neuron_count; ++n) {
            std::size_t weight_count = 0;
            if (!(file >> weight_count)) {
                throw ValidationError("Malformed neuron record: " + path);
            }
            std::vector<double> neuron(weight_count);
            for (std::size_t w = 0; w < weight_count; ++w) {
                if (!(file >> neuron[w])) {
                    throw ValidationError("Malformed weight record: " + path);
                }
            }
            layer.push_back(std::move(neuron));
        }
        weights.push_back(std::move(layer));
    }

    network.import_weights(weights);
}

void save_losses(const std::vector<double>& losses, const std::string& path, bool append) {
    std::ofstream file(path, append ? std::ios::app : std::ios::out);
    if (!file) {
        throw PathError("Cannot open loss file for writing: " + path);
    }
    file << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (double loss : losses) {
        file << loss << "\n";
    }
}

std::vector<double> load_losses(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw PathError("Loss file not found: " + path);
    }

    std::vector<double> losses;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        double value = 0.0;
        if (stream >> value) {
            losses.push_back(value);
        } else if (!line.empty()) {
            throw ValidationError("Invalid loss value in file: " + path);
        }
    }
    return losses;
}

} // namespace imagenn

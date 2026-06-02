#include "imagenn/activations.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace imagenn {

double ActivationTransparent::calc(double x) const {
    return x;
}

double ActivationTransparent::derivative(double /*x*/) const {
    return 1.0;
}

double ActivationRelu::calc(double x) const {
    return x > 0.0 ? x : 0.0;
}

double ActivationRelu::derivative(double x) const {
    return x < 0.0 ? 0.0 : 1.0;
}

double ActivationSigmoid::calc(double x) const {
    return 1.0 / (std::exp(-x) + 1.0);
}

double ActivationSigmoid::derivative(double x) const {
    const double s = calc(x);
    return s * (1.0 - s);
}

double ActivationSoftmax::calc(double /*x*/) const {
    throw std::logic_error("Softmax requires whole-layer computation");
}

double ActivationSoftmax::derivative(double /*x*/) const {
    throw std::logic_error("Softmax derivative is handled at the layer level");
}

std::vector<double> ActivationSoftmax::calc_layer(const std::vector<double>& layer_outputs) {
    if (layer_outputs.empty()) {
        throw std::invalid_argument("layer_outputs must not be empty");
    }

    std::vector<double> values = layer_outputs;
    double max_val = *std::max_element(values.begin(), values.end());

    // Большие значения масштабируются, чтобы std::exp не дал переполнения.
    if (max_val > 100.0) {
        const double scale_factor = max_val / 100.0;
        for (double& v : values) {
            v /= scale_factor;
        }
        max_val = *std::max_element(values.begin(), values.end());
    }

    double exp_sum = 0.0;
    for (double& v : values) {
        v = std::exp(v - max_val);
        exp_sum += v;
    }

    if (exp_sum == 0.0) {
        return std::vector<double>(values.size(), 1.0 / static_cast<double>(values.size()));
    }

    for (double& v : values) {
        v /= exp_sum;
    }
    return values;
}

const ActivationTransparent& transparent_activation() {
    static const ActivationTransparent instance;
    return instance;
}

const ActivationRelu& relu_activation() {
    static const ActivationRelu instance;
    return instance;
}

const ActivationSigmoid& sigmoid_activation() {
    static const ActivationSigmoid instance;
    return instance;
}

const ActivationSoftmax& softmax_activation() {
    static const ActivationSoftmax instance;
    return instance;
}

} // namespace imagenn

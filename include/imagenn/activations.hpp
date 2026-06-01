#ifndef IMAGENN_ACTIVATIONS_HPP
#define IMAGENN_ACTIVATIONS_HPP

#include <vector>

/// @file activations.hpp
/// @brief Функции активации нейронов.

namespace imagenn {

/// @brief Родительский класс функции активации.
///
/// Задаёт интерфейс: вычисление значения функции и её производной в точке.
class ActivationBase {
  public:
    virtual ~ActivationBase() = default;

    /// @brief Вычисляет значение функции активации.
    /// @param x входное значение
    /// @return выходное значение
    virtual double calc(double x) const = 0;

    /// @brief Вычисляет значение производной функции активации.
    /// @param x входное значение
    /// @return значение производной
    virtual double derivative(double x) const = 0;
};

/// @brief Прозрачная функция активации. На выходе то же значение, что и на входе.
class ActivationTransparent : public ActivationBase {
  public:
    /// @brief Возвращает входное значение без изменений.
    /// @param x входное значение
    /// @return то же значение x
    double calc(double x) const override;

    /// @brief Возвращает производную прозрачной функции.
    /// @param x входное значение
    /// @return всегда 1
    double derivative(double x) const override;
};

/// @brief Функция ReLU: y = max(0, x).
class ActivationRelu : public ActivationBase {
  public:
    /// @brief Вычисляет ReLU.
    /// @param x входное значение
    /// @return x при x > 0, иначе 0
    double calc(double x) const override;

    /// @brief Вычисляет производную ReLU.
    /// @param x входное значение
    /// @return 0 при x < 0, иначе 1
    double derivative(double x) const override;
};

/// @brief Сигмоидальная функция активации.
class ActivationSigmoid : public ActivationBase {
  public:
    /// @brief Вычисляет сигмоиду: 1 / (1 + e^(-x)).
    /// @param x входное значение
    /// @return значение функции в точке x
    double calc(double x) const override;

    /// @brief Вычисляет производную сигмоиды.
    /// @param x входное значение
    /// @return значение производной
    double derivative(double x) const override;
};

/// @brief Функция Softmax.
///
/// Softmax вычисляется для всего слоя сразу, поэтому поэлементные calc() и
/// derivative() не определены и сообщают об ошибке через исключение.
class ActivationSoftmax : public ActivationBase {
  public:
    /// @brief Не поддерживается: softmax вычисляется для всего слоя.
    /// @param x входное значение
    /// @return значение не возвращается
    /// @throws std::logic_error при любом вызове
    double calc(double x) const override;

    /// @brief Не поддерживается: производная softmax обрабатывается на уровне слоя.
    /// @param x входное значение
    /// @return значение не возвращается
    /// @throws std::logic_error при любом вызове
    double derivative(double x) const override;

    /// @brief Вычисляет softmax для всех выходов слоя.
    /// @param layer_outputs значения выходов слоя
    /// @return вектор вероятностей той же длины, сумма которых равна 1
    /// @throws std::invalid_argument если входной вектор пуст
    static std::vector<double> calc_layer(const std::vector<double>& layer_outputs);
};

} // namespace imagenn

#endif // IMAGENN_ACTIVATIONS_HPP

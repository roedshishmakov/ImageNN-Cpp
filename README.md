# ImageNN C++

Нейросеть-классификатор изображений (рукописных цифр), реализованная **с нуля на C++**
без использования ML-фреймворков. Порт проекта
[ImageNN](https://github.com/roedshishmakov/ImageNN) с Python на C++.

> ⚠️ Проект в разработке. Сейчас готов скелет сборки (CMake + библиотека + исполняемый
> файл + тесты на doctest). Полный функционал и руководство пользователя добавляются поэтапно.

## Требования

- CMake ≥ 3.16
- Компилятор с поддержкой C++17 (Apple Clang, GCC, MSVC)
- Доступ в интернет при первой конфигурации (CMake скачивает зависимости `doctest` и `stb`)

## Сборка

```bash
cmake -S . -B build
cmake --build build
```

## Запуск тестов

```bash
ctest --test-dir build --output-on-failure
```

## Запуск

```bash
./build/imagenn        # Linux/macOS
build\Debug\imagenn.exe  # Windows (MSVC)
```

## Структура проекта

```
ImageCPP/
├── CMakeLists.txt        # сборка: библиотека + exe + тесты, FetchContent зависимостей
├── include/imagenn/      # заголовочные файлы (публичный API + Doxygen-документация)
├── src/                  # реализация и точка входа (main.cpp)
├── tests/                # тесты на doctest, запускаются через ctest
├── configs/              # конфигурации сетей (.config)
├── weight_saves/         # сохранённые модели
├── loss_saves/           # история потерь
└── docs/                 # документация Doxygen (генерируется)
```

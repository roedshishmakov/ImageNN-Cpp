# ImageNN C++

Нейросеть-классификатор изображений (рукописных цифр), реализованная **с нуля на
C++** без использования ML-фреймворков. Порт проекта
[ImageNN](https://github.com/roedshishmakov/ImageNN) с Python на C++.

Сеть представлена как объектный граф нейронов и связей: forward-проход и обратное
распространение ошибки выполняются обходом по связям. Модель обучается на
пользовательском датасете, сохраняется в файл и затем используется для
классификации новых изображений.

## Возможности

- Полносвязная нейронная сеть произвольной архитектуры, задаваемой конфигурацией.
- Свёрточные слои (Conv2D), подвыборка по максимуму (MaxPooling) и выпрямление (Flatten).
- Функции активации: Transparent, ReLU, Sigmoid, Softmax.
- Выходной слой softmax с обучением по кросс-энтропии.
- Обучение с нуля, дообучение сохранённой модели.
- Сохранение и загрузка модели в собственном текстовом формате.
- График потерь в виде псевдографики в терминале.
- Классификация изображений из пользовательской папки.
- Настраиваемые пути ко всем рабочим папкам.

## Требования

- CMake 3.16 или новее.
- Компилятор с поддержкой C++17 (Apple Clang, GCC, MSVC).
- Доступ в интернет при первой конфигурации: CMake скачивает зависимости
  `doctest` и `stb` через FetchContent.

Зависимости не нужно устанавливать вручную — они собираются самим CMake.

## Сборка

Linux / macOS:

```bash
cmake -S . -B build
cmake --build build
```

Windows (Visual Studio):

```bat
cmake -S . -B build
cmake --build build --config Release
```

После сборки исполняемый файл находится в `build/imagenn`
(на Windows — `build/Release/imagenn.exe`).

## Запуск тестов

```bash
ctest --test-dir build --output-on-failure
```

## Документация кода

Документация API собрана Doxygen и лежит в папке `docs/html`. Открыть:
`docs/html/index.html`.

Пересобрать документацию (требуется Doxygen):

```bash
doxygen Doxyfile
# либо через CMake, если Doxygen найден при конфигурации:
cmake --build build --target docs
```

## Использование

Общий формат запуска:

```
imagenn <команда> [аргументы] [опции путей]
```

### Команды

| Команда | Назначение |
|---|---|
| `--train, -t <model> <config> <dataset>` | обучить сеть по файлу конфигурации |
| `--simple-train, -s <model> <epochs> <dataset>` | обучить стандартную архитектуру |
| `--fine, -f <model> <epochs> <dataset>` | дообучить сохранённую модель |
| `--load, -l <model> <images>` | классифицировать изображения сохранённой моделью |
| `--show-loss, -sl <model>` | вывести историю потерь |
| `--create-config, -c <name>` | создать шаблон конфигурации |
| `--graph, -g` | дополнительно вывести график потерь (с другими командами) |
| `--help, -h` | справка |

### Опции путей

Позволяют переопределить рабочие папки (по умолчанию относительные):

| Опция | По умолчанию |
|---|---|
| `--configs-dir <dir>` | `configs` |
| `--weights-dir <dir>` | `weight_saves` |
| `--loss-dir <dir>` | `loss_saves` |

Файлы внутри папок: модель — `<model>.nn`, конфигурация — `<model>.config`,
история потерь — `<model>.txt`.

## Сценарии использования

### Быстрое обучение и проверка

```bash
# обучить стандартную архитектуру на примерах из папки examples
imagenn --simple-train demo 10 examples

# посмотреть, как снижалась ошибка
imagenn --show-loss demo

# классифицировать изображения
imagenn --load demo examples
```

Пример вывода обучения:

```
+============================================+
| ImageNN C++  -  neural image classifier    |
+============================================+
version 1.0.0

Loaded 13 training examples
EPOCH #1/10 LOSS: 32.1078
EPOCH #2/10 LOSS: 30.5553
...
Model saved: weight_saves/demo.nn
```

Пример графика потерь (`--show-loss`):

```
Total loss
   0    32.1078 |##################################################
   1    30.5553 |###############################################
```

Пример классификации (`--load`):

```
Test file: 0_1.png  Answer: 0
Test file: 1_1.png  Answer: 1
...
```

### Обучение по своей конфигурации

```bash
# создать шаблон конфигурации configs/my_config.config
imagenn --create-config my_config

# отредактировать configs/my_config.config под свою архитектуру, затем обучить
imagenn --train my_model my_config dataset/train

# протестировать
imagenn --load my_model dataset/test
```

### Дообучение

```bash
imagenn --fine my_model 15 dataset/extra
imagenn --show-loss my_model
```

### Свои пути к папкам

```bash
imagenn --simple-train demo 10 dataset/train \
        --weights-dir /data/models --loss-dir /data/losses --configs-dir /data/configs
```

## Формат конфигурации

Текстовый файл `<name>.config` задаёт слои по порядку, затем параметры обучения.
Поддерживаются свёрточные слои, подвыборка, выпрямление и полносвязные слои.

Полносвязная сеть:

```ini
# Конфигурация нейронной сети
dense:32:relu:true:0.1
dense:32:relu:true:0.1
dense:10:softmax:false:0.1

# Параметры обучения
learning_rate=0.1
epochs=10
clip_value=5.0
use_cross_entropy=true
```

Свёрточная сеть:

```ini
conv:8:3:relu:0.1
maxpool:2
conv:16:3:relu:0.1
flatten
dense:32:relu:true:0.1
dense:10:softmax:false:0.1

learning_rate=0.05
epochs=10
clip_value=5.0
use_cross_entropy=true
```

Форматы строк слоёв:

- `dense:[size]:[activation]:[use_bias]:[random_radius]` — полносвязный слой.
- `conv:[filters]:[kernel]:[activation]:[random_radius]` — свёрточный слой (шаг 1, без дополнения краёв).
- `maxpool:[size]` — подвыборка по максимуму с окном size×size.
- `flatten` — выпрямление карт признаков в вектор.

Где:

- `activation` — `relu`, `sigmoid`, `softmax` или `transparent` (для `conv` — кроме `softmax`).
- `use_bias` — `true` или `false`.
- `random_radius` — радиус равномерной инициализации весов.

Вход — изображение 16×16 в оттенках серого, приведённое к диапазону [0, 1]. Свёрточные
слои и подвыборка должны идти до выпрямления, полносвязные — после. Если строка
`flatten` не указана, выпрямление добавляется автоматически перед первым `dense`.

## Формат датасета

- Формат файлов: PNG или JPG.
- Размер изображений: любой (приводится к 16×16).
- Имя файла: `<метка>_<идентификатор>.<расширение>`, где `<метка>` — цифра класса
  0–9.

Пример:

```
dataset/
├── 0_image1.png
├── 1_image1.png
├── 2_image1.png
└── ...
```

## Структура проекта

```
ImageCPP/
├── CMakeLists.txt        сборка: библиотека + исполняемый файл + тесты
├── include/imagenn/      заголовки (публичный API, документация Doxygen)
├── src/                  реализации и точка входа (main.cpp)
├── tests/                тесты doctest, запускаются через ctest
├── examples/             примеры изображений
├── configs/              конфигурации сетей
├── weight_saves/         сохранённые модели
├── loss_saves/           история потерь
└── docs/                 документация Doxygen (HTML)
```

## Обработка ошибок

Все ошибки сообщаются через исключения и обрабатываются на верхнем уровне.
При ошибке программа печатает сообщение в stderr и возвращает код 1, при успехе — 0.

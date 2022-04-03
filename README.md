# Tensorflow Lite Sinus Predictor

## Setup C++

### Compilation of Tensorflow Lite C++ Library

```shell
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
cd tensorflow_src
mkdir tflite_build && cd tflite_build
cmake ../tensorflow_src/tensorflow/lite
```

### Compilation

```shell
mkdir build && cd build
cmake .. -DTENSORFLOW_SOURCE_DIR=/path/to/tensorflow_src
cmake --build . -j
```

To export compile commands for clang-tidy or autocompletion, use:

```shell
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

## Setup Python

```shell
python -m pip install -r scripts/requirements.txt
```

## Usage C++

```shell
build/tflite-sinus-predictor
```

## Usage Python

### Create keras model

```shell
python scripts/create_keras_model.py
```

### Convert keras model to tflite

```shell
python scripts/convert_keras_to_tflite.py
```

### Run tflite model

```shell
python scripts/run_tflite_model.py
```

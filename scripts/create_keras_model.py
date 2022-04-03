import numpy as np
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = "../model.h5"
DATASET_SIZE = 250000


def _random_sin() -> tuple[float, float]:
    x = np.random.uniform(-np.pi, np.pi)
    return (x, np.sin(x))


def create_dataset(size: int) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices([_random_sin() for _ in range(size)])


def split_dataset(
    dataset: tf.data.Dataset, validation_split: float = 0.2
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    dataset_size = len(dataset)
    dataset.shuffle(buffer_size=dataset_size)
    train_ds = dataset.skip(int(validation_split * dataset_size))
    val_ds = dataset.take(int(validation_split * dataset_size))
    return (train_ds, val_ds)


def dataset_into_x_y(dataset: tf.data.Dataset) -> tuple[list[float], list[float]]:
    result_x = []
    result_y = []
    for x, y in dataset:
        result_x.append(x)
        result_y.append(y)
    return (result_x, result_y)


def train_keras_model():  # -> tuple[keras.Sequential, dict]:
    train_ds = create_dataset(DATASET_SIZE)
    model = keras.Sequential(
        [
            keras.layers.Dense(2, activation="relu", name="layer1", input_shape=(1,)),
            keras.layers.Dense(3, activation="relu", name="layer2"),
            keras.layers.Dense(4, activation="relu", name="layer3"),
            keras.layers.Dense(1, name="output"),
        ]
    )

    # compile model
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mse"],
    )
    # train model
    epochs = 10
    train_x, train_y = dataset_into_x_y(train_ds)
    history = model.fit(
        np.asarray(train_x), np.asarray(train_y), validation_split=0.2, epochs=epochs
    )
    model.predict([0.0])
    model.predict([1.0])
    model.predict([1.1])
    model.predict([-1.0])
    model.predict([-1.1])
    model.predict([-3.14])
    return (model, history.history)


def create_keras_model(out_model_path: str):
    model, _ = train_keras_model()
    model.save(out_model_path)


def main():
    create_keras_model(MODEL_PATH)


if __name__ == "__main__":
    main()

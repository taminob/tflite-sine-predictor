import tensorflow as tf

INPUT_MODEL = "../model.h5"
OUTPUT_MODEL = "../model.tflite"


def convert_keras_to_tflite(in_model_path: str, out_model_path: str):
    model = tf.keras.models.load_model(in_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(out_model_path, "wb") as file:
        file.write(tflite_model)


def main():
    convert_keras_to_tflite(INPUT_MODEL, OUTPUT_MODEL)


if __name__ == "__main__":
    main()

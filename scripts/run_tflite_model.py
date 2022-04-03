try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite
import numpy as np

TFLITE_MODEL = "../model.tflite"


def create_interpreter(model_path: str) -> tflite.Interpreter:
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_model(
    interpreter: tflite.Interpreter,
    inputs: list[np.ndarray[list[list[np.float32]], np.dtype[np.float32]]] = [
        np.array([[0.0]], dtype=np.float32)
    ],
) -> list[list[float]]:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # set input
    for index in range(len(input_details)):
        interpreter.set_tensor(input_details[index]["index"], inputs[index])

    # run model
    interpreter.invoke()

    # get output
    outputs = []
    for index in range(len(output_details)):
        outputs.append(interpreter.get_tensor(output_details[index]["index"]))
    return outputs[0]


def print_result(outputs: list[list[float]]):
    print(outputs)


def run_tflite_model(model_path):
    outputs = run_model(create_interpreter(model_path))
    print_result(outputs)


def main():
    run_tflite_model(TFLITE_MODEL)


if __name__ == "__main__":
    main()

#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include <iostream>
#include <memory>

constexpr const char* MODEL_PATH = "../model.tflite";

int main()
{
	auto model = tflite::FlatBufferModel::BuildFromFile(MODEL_PATH);
	if(model == nullptr)
	{
		std::cerr << "Failed to load model (" << MODEL_PATH << ")\n";
		return -1;
	}

	std::unique_ptr<tflite::Interpreter> interpreter;
	tflite::ops::builtin::BuiltinOpResolver resolver;
	if(tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk)
		return 2;
	if(interpreter->AllocateTensors() != kTfLiteOk)
		return 3;

	auto input = interpreter->input_tensor(0);
	for(auto i : interpreter->inputs())
		*interpreter->typed_tensor<float>(i) = 0.0;

	interpreter->Invoke();

	for(auto i : interpreter->outputs())
		std::cout << *interpreter->typed_tensor<float>(i) << std::endl;
}

#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "arguments.h"
#include <iostream>
#include <memory>

int main(int argc, char* argv[])
{
	auto args = arguments(argc, argv);
	auto model = tflite::FlatBufferModel::BuildFromFile(args.model_path.c_str());
	if(model == nullptr)
	{
		std::cerr << "Failed to load model (" << args.model_path << ")\n";
		return 1;
	}

	std::unique_ptr<tflite::Interpreter> interpreter;
	tflite::ops::builtin::BuiltinOpResolver resolver;
	if(tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk)
		return 2;
	if(interpreter->AllocateTensors() != kTfLiteOk)
		return 3;

	for(auto i : interpreter->inputs())
		*interpreter->typed_tensor<float>(i) = args.input;

	interpreter->Invoke();

	for(auto i : interpreter->outputs())
		std::cout << *interpreter->typed_tensor<float>(i) << std::endl;
}

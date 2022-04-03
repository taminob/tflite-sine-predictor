#ifndef TFLITE_RUNNER_H
#define TFLITE_RUNNER_H

#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/kernels/register.h"

class tflite_runner
{
	std::unique_ptr<tflite::FlatBufferModel> model;
	std::unique_ptr<tflite::Interpreter> interpreter;

public:
	tflite_runner(std::string_view model_path) :
	 model{tflite::FlatBufferModel::BuildFromFile(model_path.data())}
	{
		if(model == nullptr)
			throw std::invalid_argument{std::string{"Failed to load model ("} + model_path.data() + ")"};
		tflite::ops::builtin::BuiltinOpResolver resolver;
		if(tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk)
			throw std::runtime_error{"internal error: failed to build interpreter"};
		if(interpreter->AllocateTensors() != kTfLiteOk)
			throw std::runtime_error{"internal error: unable to allocate tensors"};
	}

	std::vector<float> run(const std::vector<float>& inputs)
	{
		if(inputs.size() != interpreter->inputs().size())
			throw std::invalid_argument{"input size does not match input of model"};
		for(int i = 0; i < inputs.size(); ++i)
			*interpreter->typed_tensor<float>(interpreter->inputs()[i]) = inputs[i];

		interpreter->Invoke();

		std::vector<float> result;
		for(auto i : interpreter->outputs())
			result.push_back(*interpreter->typed_tensor<float>(i));
		return result;
	}
};

#endif // TFLITE_RUNNER_H

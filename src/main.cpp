#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "arguments.h"
#include "tflite_runner.h"
#include <iostream>
#include <memory>

int main(int argc, char* argv[])
{
	auto args = arguments(argc, argv);
	auto runner = tflite_runner(args.model_path);
	for(auto output : runner.run({args.input}))
		std::cout << "Input: " << args.input << "\nOutput: " << output << std::endl;
}

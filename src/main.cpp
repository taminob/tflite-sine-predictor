#include "arguments.h"
#include "tflite_runner.h"
#include <iostream>

int main(int argc, char* argv[])
{
	try
	{
		auto args = arguments(argc, argv);
		auto runner = tflite_runner(args.model_path);
		for(auto output : runner.run({args.input}))
			std::cout << "Input: " << args.input << "\nOutput: " << output << std::endl;
	}
	catch(const std::invalid_argument& exc)
	{
		std::cerr << exc.what() << '\n';
	}
	catch(const std::exception& exc)
	{
		std::cerr << "internal error: " << exc.what() << '\n';
	}
}

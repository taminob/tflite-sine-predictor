#ifndef ARGUMENTS_H
#define ARGUMENTS_H

#include <iostream>
#include <string>

class arguments;

std::ostream& operator<<(std::ostream& os, const arguments&)
{
	os << "--model, -m\t<string>\tSpecifies the path to the tflite model\n";
	os << "--input, -i\t<float>\tSpecifies the input value for the model\n";
	return os;
}

struct arguments
{
	std::string model_path = "model.tflite";
	float input = 0.0;

	template<typename T, typename Func>
	int get_next_argument(
	 int argc, char* argv[], int i, T& arg, Func&& converter)
	{
		if(++i >= argc)
		{
			std::cerr << "No argument for " << argv[i] << " specified!\n";
			exit(-1);
		}
		arg = converter(argv[i]);
		return i;
	}

	arguments(int argc, char* argv[])
	{
		for(int i = 1; i < argc; ++i)
		{
			auto arg = std::string(argv[i]);
			if(arg == "--help")
				this->print_help();
			else if(arg == "--model" || arg == "-m")
				i = this->get_next_argument(argc, argv, i, model_path, [](const char* t)
				 { return t; });
			else if(arg == "--input" || arg == "-i")
				i = this->get_next_argument(argc, argv, i, input, [](const char* t)
				 { return std::stof(t); });
		}
	}

	[[noreturn]] void print_help()
	{
		std::cout << "Help: \n"
				  << *this << std::endl;
		exit(0);
	}
};

#endif // ARGUMENTS_H

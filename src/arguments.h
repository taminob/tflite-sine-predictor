#ifndef ARGUMENTS_H
#define ARGUMENTS_H

#include <array>
#include <iostream>
#include <string_view>
#include <tuple>
#include <variant>

struct arguments
{
	std::string model_path = "model.tflite";
	float input = 0.0;

	constexpr auto argument_list()
	{
		return std::array{
		 std::tuple{"--model", "-m", std::variant<std::string*, float*>(&model_path),
		  "Specifies the path to the tflite model"},
		 std::tuple{"--input", "-i", std::variant<std::string*, float*>(&input),
		  "Specifies the input value for the model"},
		};
	}

	template<typename T, typename Func>
	int get_next_argument(
	 int argc, char* argv[], int i, T& arg, Func&& converter)
	{
		if(i + 1 >= argc)
		{
			std::cerr << "No argument specified for " << argv[i] << "!\n";
			exit(-1);
		}
		arg = converter(argv[++i]);
		return i;
	}

	struct helpers
	{
		static inline float string_to_float(std::string_view s)
		{
			float result;
			bool is_valid;
			try
			{
				std::size_t converted_chars;
				result = std::stof(s.data(), &converted_chars);
				is_valid = converted_chars == s.size();
			}
			catch(const std::invalid_argument&)
			{
				is_valid = false;
			}
			catch(const std::out_of_range&)
			{
				is_valid = false;
			}
			if(!is_valid)
			{
				std::cerr << s << " is not a valid floating point number!\n";
				exit(-1);
			}
			return result;
		}

		static inline std::string string_to_string(std::string_view s)
		{
			return std::string(s);
		}
	};

	arguments(int argc, char* argv[])
	{
		for(int i = 1; i < argc; ++i)
		{
			auto arg = std::string_view(argv[i]);
			if(arg == "--help")
				this->print_help();
			else
				for(auto& a : argument_list())
				{
					if(arg == std::get<0>(a) || arg == std::get<1>(a))
					{
						auto v = std::get<2>(a);
						if(float** f = std::get_if<float*>(&v))
							i = get_next_argument(argc, argv, i, **f, helpers::string_to_float);
						else if(std::string** s = std::get_if<std::string*>(&v))
							i = get_next_argument(argc, argv, i, **s, helpers::string_to_string);
						break;
					}
				}
		}
	}

	[[noreturn]] void print_help()
	{
		auto get_type_string = [](const auto& a) -> std::string_view
		{
			if(std::holds_alternative<float*>(a))
				return "<float> ";
			if(std::holds_alternative<std::string*>(a))
				return "<string>";
			return "        ";
		};
		std::cout << "Help: \n";
		for(const auto& a : argument_list())
			std::cout << std::get<0>(a) << '\t'
					  << std::get<1>(a) << '\t'
					  << get_type_string(std::get<2>(a)) << '\t'
					  << std::get<3>(a) << '\t' << std::endl;
		exit(0);
	}
};

#endif // ARGUMENTS_H

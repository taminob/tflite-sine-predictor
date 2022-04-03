cmake_minimum_required(VERSION 3.16)
project(tflite-sinus-predictor C CXX)

set(TENSORFLOW_SOURCE_DIR
  "/home/me/software/source/tensorflow_src" CACHE PATH
  "Directory that contains the TensorFlow project")
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

add_executable(tflite-sinus-predictor src/main.cpp)
target_compile_features(tflite-sinus-predictor PRIVATE cxx_std_17)
target_link_libraries(tflite-sinus-predictor tensorflow-lite)

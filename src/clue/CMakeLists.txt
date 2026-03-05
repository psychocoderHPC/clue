# main: this will build the native C++ implementation of CLUE and its
#       corresponding native CUDA one. Which one to use must be selected at
#       runtime via a flag.

if(CMAKE_CUDA_COMPILER)
  set_source_files_properties(${PROJECT_SOURCE_DIR}/src/main.cpp PROPERTIES LANGUAGE CUDA)
  add_executable(main ${PROJECT_SOURCE_DIR}/src/main.cpp)

  target_compile_definitions(main PRIVATE)
  target_compile_options(main PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:
          -Xcompiler;
          -m64;
          -expt-relaxed-constexpr;
          -w;
          >
  )

  target_link_libraries(main PRIVATE CLUELib)

  target_include_directories(main PUBLIC
          $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  )
endif(CMAKE_CUDA_COMPILER)

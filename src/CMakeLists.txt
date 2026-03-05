if(CMAKE_CUDA_COMPILER)
  # Native C++ CLUE and native CUDA CLUE
  add_subdirectory(clue)
endif(CMAKE_CUDA_COMPILER)

# Native C++ CLUE and CLUE using ALPAKA
add_subdirectory(clue_alpaka)

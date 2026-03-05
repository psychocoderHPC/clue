# mainAlpaka: this will build the native C++ implementation of CLUE and its
#       corresponding one built using ALPAKA. Which one to use must be
#       selected at runtime via a flag.

add_executable(mainAlpaka ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_compile_definitions(mainAlpaka PRIVATE USE_ALPAKA)
target_compile_options(mainAlpaka PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
                      -Xcompiler;
                      -m64;
                      -expt-relaxed-constexpr;
                      -w;
>
)
target_link_libraries(mainAlpaka PRIVATE CLUELib)
target_link_libraries(mainAlpaka PRIVATE alpaka::alpaka)

target_include_directories(mainAlpaka PUBLIC
  $<BUILD_INTERFACE:${ALPAKA_DIR}/include>
  PUBLIC
  $<BUILD_INTERFACE:${ALPAKA_DIR}/alpaka/include>
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
alpaka_finalize(mainAlpaka)

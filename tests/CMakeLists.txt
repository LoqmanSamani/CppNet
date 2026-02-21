cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------
# CppNet Test Suite
# ---------------------------------------------------------

# Find required packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Helper function to add a CppNet test
function(cppnet_add_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE CppNet Eigen3::Eigen)
    target_include_directories(${TEST_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${TEST_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(USE_CUDA)
        target_link_libraries(${TEST_NAME} PRIVATE CUDA::cudart CUDA::cublas)
        target_compile_definitions(${TEST_NAME} PRIVATE USE_CUDA)
    endif()
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# ---------------------------------------------------------
# Layer Tests
# ---------------------------------------------------------
cppnet_add_test(test_linear layers/test_linear.cpp)
cppnet_add_test(test_conv2d layers/test_conv2d.cpp)
cppnet_add_test(test_flatten layers/test_flatten.cpp)
cppnet_add_test(test_max_pool2d layers/test_max_pool2d.cpp)
cppnet_add_test(test_rnn layers/test_rnn.cpp)
cppnet_add_test(test_attention layers/test_attention.cpp)

# ---------------------------------------------------------
# Activation Tests
# ---------------------------------------------------------
cppnet_add_test(test_relu activations/test_relu.cpp)
cppnet_add_test(test_sigmoid activations/test_sigmoid.cpp)
cppnet_add_test(test_softmax activations/test_softmax.cpp)
cppnet_add_test(test_tanh activations/test_tanh.cpp)
cppnet_add_test(test_leacky_relu activations/test_leacky_relu.cpp)

# ---------------------------------------------------------
# Loss Tests
# ---------------------------------------------------------
cppnet_add_test(test_binary_cross_entropy losses/test_binary_cross_entropy.cpp)
cppnet_add_test(test_categorical_cross_entropy losses/test_categorical_cross_entropy.cpp)
cppnet_add_test(test_mse losses/test_mse.cpp)
cppnet_add_test(test_mae losses/test_mae.cpp)
cppnet_add_test(test_huber losses/test_huber.cpp)

# ---------------------------------------------------------
# Optimizer Tests
# ---------------------------------------------------------
cppnet_add_test(test_sgd optimizers/test_sgd.cpp)
cppnet_add_test(test_adam optimizers/test_adam.cpp)
cppnet_add_test(test_momentum optimizers/test_momentum.cpp)
cppnet_add_test(test_adagrad optimizers/test_adagrad.cpp)
cppnet_add_test(test_mrs_prop optimizers/test_mrs_prop.cpp)

# ---------------------------------------------------------
# GPU Kernel Tests (only if CUDA is available)
# ---------------------------------------------------------
if(USE_CUDA)
    add_executable(test_matmul_gpu kernels/test_matmul_gpu.cu)
    target_link_libraries(test_matmul_gpu PRIVATE CppNet Eigen3::Eigen CUDA::cudart CUDA::cublas)
    target_include_directories(test_matmul_gpu PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_compile_definitions(test_matmul_gpu PRIVATE USE_CUDA)
    add_test(NAME test_matmul_gpu COMMAND test_matmul_gpu)
endif()

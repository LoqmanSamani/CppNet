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
cppnet_add_test(test_batch_norm layers/test_batch_norm.cpp)
cppnet_add_test(test_dropout layers/test_dropout.cpp)
cppnet_add_test(test_embedding layers/test_embedding.cpp)
cppnet_add_test(test_global_pool layers/test_global_pool.cpp)
cppnet_add_test(test_gru layers/test_gru.cpp)
cppnet_add_test(test_lstm layers/test_lstm.cpp)
cppnet_add_test(test_residual layers/test_residual.cpp)

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
cppnet_add_test(test_softmax_cross_entropy losses/test_softmax_cross_entropy.cpp)

# ---------------------------------------------------------
# Optimizer Tests
# ---------------------------------------------------------
cppnet_add_test(test_sgd optimizers/test_sgd.cpp)
cppnet_add_test(test_adam optimizers/test_adam.cpp)
cppnet_add_test(test_momentum optimizers/test_momentum.cpp)
cppnet_add_test(test_adagrad optimizers/test_adagrad.cpp)
cppnet_add_test(test_mrs_prop optimizers/test_mrs_prop.cpp)

# ---------------------------------------------------------
# Metric Tests
# ---------------------------------------------------------
cppnet_add_test(test_metrics metrics/test_metrics.cpp)

# ---------------------------------------------------------
# Regularization Tests
# ---------------------------------------------------------
cppnet_add_test(test_regularizations regularizations/test_regularizations.cpp)

# ---------------------------------------------------------
# Utility Tests
# ---------------------------------------------------------
cppnet_add_test(test_callbacks utils/test_callbacks.cpp)
cppnet_add_test(test_dataloader utils/test_dataloader.cpp)
cppnet_add_test(test_elapsed_time utils/test_elapsed_time.cpp)
cppnet_add_test(test_gradient_clip utils/test_gradient_clip.cpp)
cppnet_add_test(test_init utils/test_init.cpp)
cppnet_add_test(test_schedulers utils/test_schedulers.cpp)
cppnet_add_test(test_utils utils/test_utils.cpp)

# ---------------------------------------------------------
# Model Tests
# ---------------------------------------------------------
cppnet_add_test(test_models models/test_models.cpp)

# ---------------------------------------------------------
# Visualization Tests
# ---------------------------------------------------------
cppnet_add_test(test_visualizations visualizations/test_visualizations.cpp)

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

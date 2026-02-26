# =============================================================================
# CppNet Examples
# =============================================================================

function(cppnet_add_example EXAMPLE_NAME EXAMPLE_SOURCE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    target_link_libraries(${EXAMPLE_NAME} PRIVATE CppNet)
endfunction()

# Deep learning model examples
cppnet_add_example(mlp_classification           mlp_classification.cpp)
cppnet_add_example(cnn_image_classification     cnn_image_classification.cpp)
cppnet_add_example(rnn_sequence_prediction      rnn_sequence_prediction.cpp)
cppnet_add_example(transformer_classifier       transformer_classifier.cpp)
cppnet_add_example(resnet_classifier            resnet_classifier.cpp)
cppnet_add_example(mlp_benchmark                mlp_benchmark.cpp)
cppnet_add_example(cnn_benchmark                cnn_benckmark.cpp)
cppnet_add_example(sequence_benchmark           sequence_benckmark.cpp)

# mlp_benchmark uses matplotlib-cpp, which requires Python3 + NumPy
target_include_directories(mlp_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mlp_benchmark PRIVATE Python3::Python Python3::NumPy)
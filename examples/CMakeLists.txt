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

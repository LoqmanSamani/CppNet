Eigen::Tensor<double, 3> MultiHeadAttention::forward(Eigen::Tensor<double, 3>& X, Eigen::Tensor<double, 3>& Y, bool apply_mask)
        {
            int batch_size = X.dimension(0);
            int num_tokens = X.dimension(1);
            int in_size = X.dimension(2);
            
            if (in_size != in_size_)
            {
                throw std::runtime_error("Multi-Head Attention: Input dimension mismatch.");
            } 

            Eigen::Tensor<double, 4> Q;
            Eigen::Tensor<double, 4> K; 
            Eigen::Tensor<double, 4> V;
            Eigen::array<Eigen::Index, 4> reshape_dims = {batch_size, num_tokens, num_heads_, head_size_};
            Eigen::array<int, 4> shuffle_dims = {0, 2, 1, 3}; // (batch, heads, tokens, head_dim)
            Eigen::array<int, 4> shuffle_dims_transpose = {0, 1, 3, 2}; // (batch, heads, head_dim, tokens)

            if (Y.size() != 0)
            {
                // Cross-attention case: flatten both x and y
                Eigen::Tensor<double, 2> fx = f_.forward(X);
                Eigen::Tensor<double, 2> fy = f_.forward(Y);
                
                // Cache inputs for backward pass
                X_cache_ = fx;
                Y_cache_ = fy;

                // Linear projections - Q from Y, K and V from X for cross-attention
                Eigen::Tensor<double, 2> fq = dense_forward(fy, Wq_, bq_); // [batch*seq_y, embed]
                Eigen::Tensor<double, 2> fk = dense_forward(fx, Wk_, bk_); // [batch*seq_x, embed]
                Eigen::Tensor<double, 2> fv = dense_forward(fx, Wv_, bv_); // [batch*seq_x, embed]

                // Reshape to 4D tensors (batch, tokens, heads, head_dim)
                Eigen::Tensor<double, 4> Q_reshaped = fq.reshape(reshape_dims);
                Eigen::Tensor<double, 4> K_reshaped = fk.reshape(reshape_dims);
                Eigen::Tensor<double, 4> V_reshaped = fv.reshape(reshape_dims);
                
                // Transpose to (batch, heads, tokens, head_dim)
                Q = Q_reshaped.shuffle(shuffle_dims);
                K = K_reshaped.shuffle(shuffle_dims);
                V = V_reshaped.shuffle(shuffle_dims);    
            }
            else
            {
                // Self-attention case: flatten x
                Eigen::Tensor<double, 2> fx = f_.forward(X);
                
                // Cache input for backward pass
                in_cache_ = fx;
                
                // Linear projections - all from the same input
                Eigen::Tensor<double, 2> fq = dense_forward(fx, Wq_, bq_); // [batch*seq, embed]
                Eigen::Tensor<double, 2> fk = dense_forward(fx, Wk_, bk_); // [batch*seq, embed]
                Eigen::Tensor<double, 2> fv = dense_forward(fx, Wv_, bv_); // [batch*seq, embed]

                // Reshape to 4D tensors (batch, tokens, heads, head_dim)
                Eigen::Tensor<double, 4> Q_reshaped = fq.reshape(reshape_dims);
                Eigen::Tensor<double, 4> K_reshaped = fk.reshape(reshape_dims);
                Eigen::Tensor<double, 4> V_reshaped = fv.reshape(reshape_dims);
            
                // Transpose to (batch, heads, tokens, head_dim)
                Q = Q_reshaped.shuffle(shuffle_dims);
                K = K_reshaped.shuffle(shuffle_dims);
                V = V_reshaped.shuffle(shuffle_dims); 
            }

            // Cache Q, K, V for backward pass
            Q_cache_ = Q;
            K_cache_ = K;
            V_cache_ = V;

            // Transpose K for matrix multiplication: (batch, heads, head_dim, tokens)
            Eigen::Tensor<double, 4> tK = K.shuffle(shuffle_dims_transpose);
            
            // CORRECTED: Reshape to 3D for proper batch matrix multiplication
            // Q: [batch_size, num_heads, num_tokens, head_size] -> [batch_size * num_heads, num_tokens, head_size]
            // tK: [batch_size, num_heads, head_size, num_tokens] -> [batch_size * num_heads, head_size, num_tokens]
            Eigen::array<Eigen::Index, 3> reshape_Q = {batch_size * num_heads_, num_tokens, head_size_};
            Eigen::array<Eigen::Index, 3> reshape_tK = {batch_size * num_heads_, head_size_, num_tokens};
            
            Eigen::Tensor<double, 3> Q_reshaped = Q.reshape(reshape_Q);
            Eigen::Tensor<double, 3> tK_reshaped = tK.reshape(reshape_tK);

            // Compute attention scores: Q @ K^T
            // [batch*heads, tokens, head_dim] × [batch*heads, head_dim, tokens] -> [batch*heads, tokens, tokens]
            Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(2, 1)};
            Eigen::Tensor<double, 4> att_scores_4d = Q_reshaped.contract(tK_reshaped, product_dims);

            std::cout << att_scores_4d.size() << att_scores_4d.dimensions() << std::endl;
            
            // Reshape back to 4D: [batch, heads, tokens, tokens]
            Eigen::array<Eigen::Index, 4> attention_shape = {batch_size, num_heads_, num_tokens, num_tokens};
            Eigen::Tensor<double, 4> att_scores = att_scores_4d.reshape(attention_shape);
            
            // Scale by sqrt(head_size) for stability
            double scale_factor = 1.0 / std::sqrt(static_cast<double>(head_size_));
            att_scores = att_scores * scale_factor;

            // Apply causal mask if requested
            if (apply_mask)
            {
                apply_causal_mask(att_scores, mask_, num_tokens);
            }

            // Apply softmax along the last dimension (tokens) to get attention weights
            int total_batch_heads = batch_size * num_heads_;
            Eigen::array<Eigen::Index, 2> flatten_dims = {total_batch_heads * num_tokens, num_tokens};
            Eigen::Tensor<double, 2> att_scores_2d = att_scores.reshape(flatten_dims);
            CppNet::Activations::SoftMax softmax;
            Eigen::Tensor<double, 2> attention_weights_2d = softmax.forward(att_scores_2d);
            Eigen::array<Eigen::Index, 4> unflatten_dims = {batch_size, num_heads_, num_tokens, num_tokens};
            Eigen::Tensor<double, 4> attention_weights = attention_weights_2d.reshape(unflatten_dims);
            
            // Cache attention weights for backward pass
            attention_weights_cache_ = attention_weights;

            // TODO: apply dropout

            // Compute context vector: attention_weights @ V
            // Same approach: reshape to 3D, contract, then reshape back
            Eigen::array<Eigen::Index, 3> reshape_weights = {batch_size * num_heads_, num_tokens, num_tokens};
            Eigen::array<Eigen::Index, 3> reshape_V = {batch_size * num_heads_, num_tokens, head_size_};
            
            Eigen::Tensor<double, 3> weights_reshaped = attention_weights.reshape(reshape_weights);
            Eigen::Tensor<double, 3> V_reshaped = V.reshape(reshape_V);
            
            // [batch*heads, tokens, tokens] × [batch*heads, tokens, head_dim] -> [batch*heads, tokens, head_dim]
            Eigen::array<Eigen::IndexPair<int>, 1> context_product_dims = {Eigen::IndexPair<int>(2, 1)};
            Eigen::Tensor<double, 3> context_3d = weights_reshaped.contract(V_reshaped, context_product_dims);
            
            // Reshape back to 4D: [batch, heads, tokens, head_dim]
            Eigen::array<Eigen::Index, 4> context_4d_shape = {batch_size, num_heads_, num_tokens, head_size_};
            Eigen::Tensor<double, 4> context_vector = context_3d.reshape(context_4d_shape);
            
            // Transpose back to (batch, tokens, heads, head_dim)
            Eigen::array<int, 4> output_shuffle_dims = {0, 2, 1, 3};
            Eigen::Tensor<double, 4> context_transposed = context_vector.shuffle(output_shuffle_dims);
            
            // Reshape to (batch, tokens, out_size) - concatenate heads
            Eigen::array<Eigen::Index, 3> final_reshape_dims = {batch_size, num_tokens, out_size_};
            Eigen::Tensor<double, 3> output = context_transposed.reshape(final_reshape_dims);
            
            return output;
        }
        
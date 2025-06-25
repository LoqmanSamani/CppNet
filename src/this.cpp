/************************************** Flatten *******************************************/ 

        Flatten::Flatten(int start_dim, int end_dim, std::string layer_name) :
            start_dim_(start_dim), end_dim_(end_dim), layer_name_(layer_name),
            in_size_(0), out_size_(0), input_rank_(0) 
        {
            if (start_dim < 0) {
                throw std::invalid_argument("Flatten: start_dim must be non-negative");
            }
        }

        Eigen::Tensor<double, 2> Flatten::forward(const Eigen::Tensor<double, 4>& X) {
            input_rank_ = 4;
            if (X.size() == 0) {
                throw std::runtime_error("Flatten: Empty input tensor");
            }

            // Store input shape
            in_shape_.resize(4);
            in_size_ = 1;
            for (int i = 0; i < 4; ++i) {
                in_shape_[i] = X.dimension(i);
                in_size_ *= X.dimension(i);
            }

            // Resolve end_dim
            int resolved_end_dim = end_dim_ < 0 ? 3 : end_dim_;
            if (start_dim_ >= 4 || resolved_end_dim >= 4 || start_dim_ > resolved_end_dim) {
                throw std::invalid_argument("Flatten: Invalid start_dim or end_dim for 4D tensor");
            }

            // Common case: keep batch dimension, flatten the rest
            if (start_dim_ == 1 && resolved_end_dim == 3) {
                int batch_size = X.dimension(0);
                int feature_size = X.dimension(1) * X.dimension(2) * X.dimension(3);
                out_size_ = feature_size;
                
                Eigen::array<int, 2> out_dims = {batch_size, feature_size};
                return X.reshape(out_dims);
            }
            
            // General case
            int first_dim = 1;
            for (int i = 0; i < start_dim_; ++i) {
                first_dim *= X.dimension(i);
            }
            
            int second_dim = 1;
            for (int i = start_dim_; i <= resolved_end_dim; ++i) {
                second_dim *= X.dimension(i);
            }
            
            for (int i = resolved_end_dim + 1; i < 4; ++i) {
                second_dim *= X.dimension(i);
            }
            
            out_size_ = second_dim;
            Eigen::array<int, 2> out_dims = {first_dim, second_dim};
            return X.reshape(out_dims);
        }

        Eigen::Tensor<double, 2> Flatten::forward(const Eigen::Tensor<double, 3>& X) {
            input_rank_ = 3;
            // Similar implementation for 3D tensors
            in_shape_.resize(3);
            in_size_ = 1;
            for (int i = 0; i < 3; ++i) {
                in_shape_[i] = X.dimension(i);
                in_size_ *= X.dimension(i);
            }

            if (start_dim_ == 1) {
                int batch_size = X.dimension(0);
                int feature_size = X.dimension(1) * X.dimension(2);
                out_size_ = feature_size;
                
                Eigen::array<int, 2> out_dims = {batch_size, feature_size};
                return X.reshape(out_dims);
            }
            
            // Default: flatten all dimensions
            Eigen::array<int, 2> out_dims = {1, static_cast<int>(X.size())};
            return X.reshape(out_dims);
        }

        Eigen::Tensor<double, 2> Flatten::forward(const Eigen::Tensor<double, 2>& X) {
            input_rank_ = 2;
            in_shape_.resize(2);
            for (int i = 0; i < 2; ++i) {
                in_shape_[i] = X.dimension(i);
            }
            in_size_ = X.size();
            out_size_ = X.size();
            return X; // Already 2D
        }

        Eigen::Tensor<double, 4> Flatten::backward4D(const Eigen::Tensor<double, 2>& dY) {
            if (input_rank_ != 4) {
                throw std::runtime_error("Flatten: backward4D called but input was not 4D");
            }
            if (in_shape_.size() != 4) {
                throw std::runtime_error("Flatten: Input shape not set for 4D tensor");
            }

            Eigen::array<int, 4> in_dims;
            for (int i = 0; i < 4; ++i) {
                in_dims[i] = in_shape_[i];
            }
            return dY.reshape(in_dims);
        }

        Eigen::Tensor<double, 3> Flatten::backward3D(const Eigen::Tensor<double, 2>& dY) {
            if (input_rank_ != 3) {
                throw std::runtime_error("Flatten: backward3D called but input was not 3D");
            }
            if (in_shape_.size() != 3) {
                throw std::runtime_error("Flatten: Input shape not set for 3D tensor");
            }

            Eigen::array<int, 3> in_dims;
            for (int i = 0; i < 3; ++i) {
                in_dims[i] = in_shape_[i];
            }
            return dY.reshape(in_dims);
        }

        Eigen::Tensor<double, 2> Flatten::backward2D(const Eigen::Tensor<double, 2>& dY) {
            return dY; // Already 2D
        }

        /************************************** Multi-Head Attention *******************************************/ 
        MultiHeadAttention::MultiHeadAttention
            (
                int in_size,
                int out_size,
                int num_heads = 8,
                int context_length = 512,
                double dropout_rate = 0.2,
                std::string layer_name = "Multi-Head Attention",
                bool trainable = true,
                bool qkv_bias = false


            ) :
            in_size_(in_size), out_size_(out_size),
            num_heads_(num_heads), context_length_(context_length),
            dropout_rate_(dropout_rate), layer_name_(layer_name),
            trainable_(trainable), qkv_bias_(qkv_bias)
            {
                head_size_ = in_size_ / num_heads_;
                mask_ = create_causal_mask(context_length_);
                bool is_divisible = in_size_ % num_heads_;
                if (!is_divisible)
                {
                    throw std::runtime_error("Multi-Head Attention: Input dimension must be divisible by the number of heads.");
                }   
            }
        Eigen::Tensor<double, 2> MultiHeadAttention::dense_forward(
            const Eigen::Tensor<double, 2>& X,
            Eigen::Tensor<double, 2>& W, 
            Eigen::Tensor<double, 1>& b)
        {
            Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
            Eigen::Tensor<double, 2> output = X.contract(W, product_dims);

            if (qkv_bias_) {
                
                Eigen::array<Eigen::Index, 2> broadcast_dims({X.dimension(0), 1});
                Eigen::Tensor<double, 2> bias_broadcasted = b.reshape(Eigen::array<Eigen::Index, 2>({1, b.dimension(0)})).broadcast(broadcast_dims);
                output = output + bias_broadcasted;
            }
            
            return output;
        }
        Eigen::Tensor<double, 2> MultiHeadAttention::dense_backward(
            const Eigen::Tensor<double, 2>& grad_out,
            Eigen::Tensor<double, 2>in_cache,
            Eigen::Tensor<double, 2> weights,
            Eigen::Tensor<double, 2>& grad_weights,
            Eigen::Tensor<double, 1>& grad_biases)
        {
            if (trainable_) {
                Eigen::array<int, 2> transpose_dims({1, 0});
                Eigen::Tensor<double, 2> X_transposed = in_cache.shuffle(transpose_dims);
                
                Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
                grad_weights = X_transposed.contract(grad_out, product_dims);
                
                if (qkv_bias_) 
                {
                    Eigen::array<int, 1> batch_dim({0});
                    grad_biases = grad_out.sum(batch_dim);
                }
            }
            Eigen::array<int, 2> transpose_dims({1, 0});
            Eigen::Tensor<double, 2> weights_transposed = weights.shuffle(transpose_dims);
            
            Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
            Eigen::Tensor<double, 2> grad_input = grad_out.contract(weights_transposed, product_dims);
            
            return grad_input;
        }

        void MultiHeadAttention::init_params_and_grads()
        {
            std::random_device rd;
            std::mt19937 gen(rd());

            double scale = std::sqrt(6.0 / (in_size_ + out_size_));
            std::uniform_real_distribution<> dis(-scale, scale);

            // initialize weight-tensors
            Wq_ = Eigen::Tensor<double, 2>(in_size_, out_size_);
            Wk_ = Eigen::Tensor<double, 2>(in_size_, out_size_);
            Wv_ = Eigen::Tensor<double, 2>(in_size_, out_size_);

            for (int i = 0; i < in_size_; ++i) 
            {
                for (int j = 0; j < out_size_; ++j) 
                {
                    Wq_(i, j) = dis(gen);
                    Wk_(i, j) = dis(gen);
                    Wv_(i, j) = dis(gen);
                }
            }

            // initialize weight-gradient matrices with zero
            grad_Wq_ = Eigen::Tensor<double, 2>(in_size_, out_size_);
            grad_Wk_ = Eigen::Tensor<double, 2>(in_size_, out_size_);
            grad_Wv_ = Eigen::Tensor<double, 2>(in_size_, out_size_);

            grad_Wq_.setZero();
            grad_Wk_.setZero();
            grad_Wv_.setZero();

            if (qkv_bias_)
            {
                // initialize biases with zero, if bias_ is true.
                bq_ = Eigen::Tensor<double, 1>(out_size_);
                bk_ = Eigen::Tensor<double, 1>(out_size_);
                bv_ = Eigen::Tensor<double, 1>(out_size_);

                bq_.setZero();
                bk_.setZero();
                bv_.setZero();

                // initialize bias-gradient matrix with zero.
                grad_bq_ = Eigen::Tensor<double, 1>(out_size_);
                grad_bk_ = Eigen::Tensor<double, 1>(out_size_);
                grad_bv_ = Eigen::Tensor<double, 1>(out_size_);

                grad_bq_.setZero(); 
                grad_bk_.setZero();
                grad_bv_.setZero(); 
            }
            else
            {
                // initialize empty biases and gradients when bias is false
                bq_ = Eigen::Tensor<double, 1>(0);
                bk_ = Eigen::Tensor<double, 1>(0);
                bv_ = Eigen::Tensor<double, 1>(0);

                grad_bq_ = Eigen::Tensor<double, 1>(0);
                grad_bk_ = Eigen::Tensor<double, 1>(0);
                grad_bv_ = Eigen::Tensor<double, 1>(0);
            }
        }

        Eigen::Tensor<bool, 2> create_causal_mask(int context_length) 
        {
            Eigen::Tensor<int, 2> row_indices(context_length, context_length);
            Eigen::Tensor<int, 2> col_indices(context_length, context_length);
            
            // create coordinate matrices
            for (int i = 0; i < context_length; ++i) {
                for (int j = 0; j < context_length; ++j) {
                    row_indices(i, j) = i;
                    col_indices(i, j) = j;
                }
            }
            
            // create mask where column > row (upper triangular)
            Eigen::Tensor<bool, 2> mask = col_indices > row_indices;
            
            return mask;
        }

        void apply_causal_mask(Eigen::Tensor<double, 4>& att_scores, const Eigen::Tensor<bool, 2>& mask, int num_tokens) 
        {
            // get dimensions from attention_scores
            int batch_size = att_scores.dimension(0);
            int num_heads = att_scores.dimension(1);
            
            // apply the mask (equivalent to masked_fill with -inf)
            for (int b = 0; b < batch_size; ++b) {
                for (int h = 0; h < num_heads; ++h) {
                    for (int i = 0; i < num_tokens; ++i) {
                        for (int j = 0; j < num_tokens; ++j) {
                            if (mask(i, j)) {
                                att_scores(b, h, i, j) = -std::numeric_limits<double>::infinity();
                            }
                        }
                    }
                }
            }
        }
        
        Eigen::Tensor<double, 3> MultiHeadAttention::forward(Eigen::Tensor<double, 3>& X, Eigen::Tensor<double, 3>& Y, bool apply_mask)
        {
            int batch_size = X.dimension(0);
            int num_tokens = X.dimension(1);
            int in_size = X.dimension(2);
            Flatten f;
            
            if (in_size != in_size_)
            {
                throw std::runtime_error("Multi-Head Attention: Input dimension mismatch.");
            } 
            
            Eigen::Tensor<double, 4> Q;
            Eigen::Tensor<double, 4> K; 
            Eigen::Tensor<double, 4> V;
            Eigen::array<Eigen::Index, 4> reshape_dims = {batch_size, num_tokens, num_heads_, head_size_};
            Eigen::array<int, 4> shuffle_dims = {0, 2, 1, 3}; // (batch, heads, tokens, head_dim)
            Eigen::array<int, 4> shuffle_dims1 = {0, 2, 3, 1}; // (batch, heads, head_dim, tokens)

            if (Y.size() != 0)
            {
                // cross-attention case: flatten both x and y
                Eigen::Tensor<double, 2> fx = f.forward(X);
                Eigen::Tensor<double, 2> fy = f.forward(Y);
                
                // cache inputs for backward pass
                X_cache_ = fx;
                Y_cache_ = fy;

                // linear projections - Q from Y, K and V from X for cross-attention
                Eigen::Tensor<double, 2> fq = dense_forward(fy, Wq_, bq_);
                Eigen::Tensor<double, 2> fk = dense_forward(fx, Wk_, bk_);
                Eigen::Tensor<double, 2> fv = dense_forward(fx, Wv_, bv_);

                // reshape to 4d tensors (batch, tokens, heads, head_dim)
                Eigen::Tensor<double, 4> Q_reshaped = fq.reshape(reshape_dims);
                Eigen::Tensor<double, 4> K_reshaped = fk.reshape(reshape_dims);
                Eigen::Tensor<double, 4> V_reshaped = fv.reshape(reshape_dims);
                
                // transpose to (batch, heads, tokens, head_dim)
                Q = Q_reshaped.shuffle(shuffle_dims);
                K = K_reshaped.shuffle(shuffle_dims);
                V = V_reshaped.shuffle(shuffle_dims);    
            }
            else
            {
                // self-attention case: flatten x
                Eigen::Tensor<double, 2> fx = f.forward(X);
                
                // cache input for backward pass
                in_cache_ = fx;
                
                // linear projections - all from the same input
                Eigen::Tensor<double, 2> fq = dense_forward(fx, Wq_, bq_);
                Eigen::Tensor<double, 2> fk = dense_forward(fx, Wk_, bk_);
                Eigen::Tensor<double, 2> fv = dense_forward(fx, Wv_, bv_);

                // reshape to 4d tensors (batch, tokens, heads, head_dim)
                Eigen::Tensor<double, 4> Q_reshaped = fq.reshape(reshape_dims);
                Eigen::Tensor<double, 4> K_reshaped = fk.reshape(reshape_dims);
                Eigen::Tensor<double, 4> V_reshaped = fv.reshape(reshape_dims);
            
                // transpose to (batch, heads, tokens, head_dim)
                Q = Q_reshaped.shuffle(shuffle_dims);
                K = K_reshaped.shuffle(shuffle_dims);
                V = V_reshaped.shuffle(shuffle_dims); 
            }

            // cache Q, K, V for backward pass
            Q_cache_ = Q;
            K_cache_ = K;
            V_cache_ = V;

            // transpose K for matrix multiplication: (batch, heads, head_dim, tokens)
            Eigen::Tensor<double, 4> tK = K.shuffle(shuffle_dims1);

            // compute attention scores: Q @ K^T
            Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(3, 2)};
            Eigen::Tensor<double, 4> att_scores = Q.contract(tK, product_dims);
            
            // scale by sqrt(head_size) for stability
            double scale_factor = 1.0 / std::sqrt(static_cast<double>(head_size_));
            att_scores = att_scores * scale_factor;

            // apply causal mask if requested
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
            
            // cache attention weights for backward pass
            attention_weights_cache_ = attention_weights;

            // TODO: apply dropout
           
            // compute context vector: attention_weights @ V
            Eigen::array<Eigen::IndexPair<int>, 1> context_product_dims = {Eigen::IndexPair<int>(3, 2)};
            Eigen::Tensor<double, 4> context_vector = attention_weights.contract(V, context_product_dims);
            
            // transpose back to (batch, tokens, heads, head_dim)
            Eigen::array<int, 4> output_shuffle_dims = {0, 2, 1, 3};
            Eigen::Tensor<double, 4> context_transposed = context_vector.shuffle(output_shuffle_dims);
            
            // reshape to (batch, tokens, out_size) - concatenate heads
            Eigen::array<Eigen::Index, 3> final_reshape_dims = {batch_size, num_tokens, out_size_};
            Eigen::Tensor<double, 3> output = context_transposed.reshape(final_reshape_dims);
            
            return output;
        }

        Eigen::Tensor<double, 3> MultiHeadAttention::backward(Eigen::Tensor<double, 3>& dA, Eigen::Tensor<double, 3>& dY)
        {
            // get dimensions from the gradient
            int batch_size = dA.dimension(0);
            int num_tokens = dA.dimension(1);
            int out_size = dA.dimension(2);
            
            if (out_size != out_size_)
            {
                throw std::runtime_error("Multi-Head Attention Backward: Output dimension mismatch.");
            }
            
            Flatten f;
            
            // reshape gradient to (batch, tokens, heads, head_dim)
            Eigen::array<Eigen::Index, 4> grad_reshape_dims = {batch_size, num_tokens, num_heads_, head_size_};
            Eigen::Tensor<double, 4> dA_reshaped = dA.reshape(grad_reshape_dims);
            
            // transpose to (batch, heads, tokens, head_dim)
            Eigen::array<int, 4> shuffle_dims = {0, 2, 1, 3};
            Eigen::Tensor<double, 4> dA_transposed = dA_reshaped.shuffle(shuffle_dims);
            
           
            // gradient w.r.t. context vector (before concatenation)
            Eigen::Tensor<double, 4> dContext = dA_transposed;
            
            // gradient w.r.t. attention weights: dContext @ V^T
            Eigen::array<int, 4> V_transpose_dims = {0, 1, 3, 2}; // (batch, heads, head_dim, tokens)
            Eigen::Tensor<double, 4> V_transposed = V_cache_.shuffle(V_transpose_dims);
            
            Eigen::array<Eigen::IndexPair<int>, 1> dAtt_product_dims = {Eigen::IndexPair<int>(3, 2)};
            Eigen::Tensor<double, 4> dAttention_weights = dContext.contract(V_transposed, dAtt_product_dims);
            
            // gradient w.r.t. V: attention_weights^T @ dContext
            Eigen::array<int, 4> att_transpose_dims = {0, 1, 3, 2}; // (batch, heads, tokens, tokens)
            Eigen::Tensor<double, 4> attention_weights_transposed = attention_weights_cache_.shuffle(att_transpose_dims);
            
            Eigen::array<Eigen::IndexPair<int>, 1> dV_product_dims = {Eigen::IndexPair<int>(2, 2)};
            Eigen::Tensor<double, 4> dV = attention_weights_transposed.contract(dContext, dV_product_dims);
            
            // gradient through softmax
            int total_batch_heads = batch_size * num_heads_;
            Eigen::array<Eigen::Index, 2> flatten_dims = {total_batch_heads * num_tokens, num_tokens};
            Eigen::Tensor<double, 2> dAttention_weights_2d = dAttention_weights.reshape(flatten_dims);
            CppNet::Activations::SoftMax softmax;
            Eigen::Tensor<double, 2> dAtt_scores_2d = softmax.backward(dAttention_weights_2d);
            Eigen::array<Eigen::Index, 4> unflatten_dims = {batch_size, num_heads_, num_tokens, num_tokens};
            Eigen::Tensor<double, 4> dAtt_scores = dAtt_scores_2d.reshape(unflatten_dims);
            
            // apply scaling factor gradient
            double scale_factor = 1.0 / std::sqrt(static_cast<double>(head_size_));
            dAtt_scores = dAtt_scores * scale_factor;
            
            // gradient w.r.t. Q: dAtt_scores @ K
            Eigen::array<Eigen::IndexPair<int>, 1> dQ_product_dims = {Eigen::IndexPair<int>(3, 3)};
            Eigen::Tensor<double, 4> dQ = dAtt_scores.contract(K_cache_, dQ_product_dims);
            
            // gradient w.r.t. K: dAtt_scores^T @ Q
            Eigen::array<int, 4> dAtt_transpose_dims = {0, 1, 3, 2}; // (batch, heads, tokens, tokens)
            Eigen::Tensor<double, 4> dAtt_scores_transposed = dAtt_scores.shuffle(dAtt_transpose_dims);
            
            Eigen::array<Eigen::IndexPair<int>, 1> dK_product_dims = {Eigen::IndexPair<int>(2, 2)};
            Eigen::Tensor<double, 4> dK = dAtt_scores_transposed.contract(Q_cache_, dK_product_dims);
            
            // transpose Q, K, V gradients back to (batch, tokens, heads, head_dim)
            Eigen::array<int, 4> output_shuffle_dims = {0, 2, 1, 3};
            Eigen::Tensor<double, 4> dQ_transposed = dQ.shuffle(output_shuffle_dims);
            Eigen::Tensor<double, 4> dK_transposed = dK.shuffle(output_shuffle_dims);
            Eigen::Tensor<double, 4> dV_transposed = dV.shuffle(output_shuffle_dims);
            
            // reshape to 3D for linear layer backward pass
            Eigen::array<Eigen::Index, 3> linear_reshape_dims = {batch_size, num_tokens, out_size_};
            Eigen::Tensor<double, 3> dQ_3d = dQ_transposed.reshape(linear_reshape_dims);
            Eigen::Tensor<double, 3> dK_3d = dK_transposed.reshape(linear_reshape_dims);
            Eigen::Tensor<double, 3> dV_3d = dV_transposed.reshape(linear_reshape_dims);
            
            // flatten to 2D for dense layer backward pass
            Eigen::Tensor<double, 2> dQ_2d = f.forward(dQ_3d);
            Eigen::Tensor<double, 2> dK_2d = f.forward(dK_3d);
            Eigen::Tensor<double, 2> dV_2d = f.forward(dV_3d);
            
            // initialize gradients
            Eigen::Tensor<double, 2> dX;
            Eigen::Tensor<double, 2> dY_output;
            
            if (dY.size() != 0) // cross-attention case
            {
                // gradient w.r.t. input X (comes from K and V)
                Eigen::Tensor<double, 2> dX_from_K = dense_backward(dK_2d, X_cache_, Wk_, grad_Wk_, grad_bk_);
                Eigen::Tensor<double, 2> dX_from_V = dense_backward(dV_2d, X_cache_, Wv_, grad_Wv_, grad_bv_);
                dX = dX_from_K + dX_from_V;
                
                // gradient w.r.t. input Y (comes from Q)
                dY_output = dense_backward(dQ_2d, Y_cache_, Wq_, grad_Wq_, grad_bq_);
                
                // reshape back to 3D
                Eigen::array<Eigen::Index, 3> output_reshape_dims = {batch_size, num_tokens, in_size_};
                Eigen::Tensor<double, 3> dX_3d = dX.reshape(output_reshape_dims);
                Eigen::Tensor<double, 3> dY_3d = dY_output.reshape(output_reshape_dims);
                
                // for cross-attention, we return gradient w.r.t. X, and dY is modified in place
                dY = dY_3d;
                return dX_3d;
            }
            else // self-attention case
            {
                // all gradients go to the same input X
                Eigen::Tensor<double, 2> dX_from_Q = dense_backward(dQ_2d, in_cache_, Wq_, grad_Wq_, grad_bq_);
                Eigen::Tensor<double, 2> dX_from_K = dense_backward(dK_2d, in_cache_, Wk_, grad_Wk_, grad_bk_);
                Eigen::Tensor<double, 2> dX_from_V = dense_backward(dV_2d, in_cache_, Wv_, grad_Wv_, grad_bv_);
                
                dX = dX_from_Q + dX_from_K + dX_from_V;
                
                // reshape back to 3D
                Eigen::array<Eigen::Index, 3> output_reshape_dims = {batch_size, num_tokens, in_size_};
                return dX.reshape(output_reshape_dims);
            }
        }
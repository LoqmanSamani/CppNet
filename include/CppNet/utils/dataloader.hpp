/**
 * @file dataloader.hpp
 * @brief Mini-batch data loader / iterator for CppNet
 *
 * Wraps a dataset (features + labels as Eigen tensors) and provides
 * an iterator interface that yields shuffled mini-batches.
 *
 * @example
 *   DataLoader loader(X, Y, /*batch_size=*/ 32, /*shuffle=*/ true);
 *   for (auto& [xb, yb] : loader) {
 *       auto out = model.forward(xb);
 *       ...
 *   }
 */

#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <utility>
#include <cstddef>

namespace CppNet
{
    namespace Utils
    {
        /**
         * @struct Batch
         * @brief A single mini-batch of (features, labels)
         */
        struct Batch
        {
            Eigen::Tensor<float, 2> features;  // [batch_size, num_features]
            Eigen::Tensor<float, 2> labels;     // [batch_size, num_labels]
        };

        /**
         * @class DataLoader
         * @brief Iterable mini-batch provider with optional shuffling
         */
        class DataLoader
        {
        public:
            /**
             * @param features   Full dataset features [N, F]
             * @param labels     Full dataset labels   [N, L]
             * @param batch_size Mini-batch size
             * @param shuffle    Shuffle indices at the beginning of each epoch
             * @param drop_last  If true, drop the last incomplete batch
             */
            DataLoader(const Eigen::Tensor<float, 2>& features,
                       const Eigen::Tensor<float, 2>& labels,
                       int batch_size,
                       bool shuffle = true,
                       bool drop_last = false);

            /// Re-shuffle indices (call at start of each epoch)
            void reset();

            /// Number of batches in one full pass
            std::size_t num_batches() const { return num_batches_; }

            /// Total number of samples
            int num_samples() const { return num_samples_; }

            int get_batch_size() const { return batch_size_; }

            // -------------------------------------------------------
            //  STL-style iterator
            // -------------------------------------------------------

            class Iterator
            {
            public:
                Iterator(DataLoader* loader, std::size_t batch_idx);

                Batch operator*() const;
                Iterator& operator++();
                bool operator!=(const Iterator& other) const;

            private:
                DataLoader* loader_;
                std::size_t batch_idx_;
            };

            Iterator begin();
            Iterator end();

        private:
            Eigen::Tensor<float, 2> features_;
            Eigen::Tensor<float, 2> labels_;
            int batch_size_;
            bool shuffle_;
            bool drop_last_;
            int num_samples_;
            int num_features_;
            int num_labels_;
            std::size_t num_batches_;

            std::vector<int> indices_;
            std::mt19937 gen_;

            /// Extract a Batch given start position in indices_
            Batch get_batch(std::size_t batch_idx) const;
        };
    }
}

#endif // DATALOADER_HPP

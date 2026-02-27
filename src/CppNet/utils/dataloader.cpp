/**
 * @file dataloader.cpp
 * @brief DataLoader / mini-batch iterator implementation
 */

#include "CppNet/utils/dataloader.hpp"
#include <stdexcept>

namespace CppNet
{
    namespace Utils
    {
        DataLoader::DataLoader(const Eigen::Tensor<float, 2>& features,
                               const Eigen::Tensor<float, 2>& labels,
                               int batch_size,
                               bool shuffle,
                               bool drop_last)
            : features_(features), labels_(labels),
              batch_size_(batch_size), shuffle_(shuffle), drop_last_(drop_last),
              gen_(static_cast<unsigned>(
                  std::chrono::steady_clock::now().time_since_epoch().count()))
        {
            num_samples_  = features_.dimension(0);
            num_features_ = features_.dimension(1);
            num_labels_   = labels_.dimension(1);

            if (labels_.dimension(0) != num_samples_)
                throw std::invalid_argument(
                    "DataLoader: features and labels must have the same number of samples");

            if (batch_size_ <= 0)
                throw std::invalid_argument("DataLoader: batch_size must be > 0");

            // compute number of batches
            if (drop_last_)
                num_batches_ = static_cast<std::size_t>(num_samples_ / batch_size_);
            else
                num_batches_ = static_cast<std::size_t>(
                    (num_samples_ + batch_size_ - 1) / batch_size_);

            // initialize index vector
            indices_.resize(num_samples_);
            for (int i = 0; i < num_samples_; ++i)
                indices_[i] = i;

            if (shuffle_)
                std::shuffle(indices_.begin(), indices_.end(), gen_);
        }

        void DataLoader::reset()
        {
            if (shuffle_)
                std::shuffle(indices_.begin(), indices_.end(), gen_);
        }

        Batch DataLoader::get_batch(std::size_t batch_idx) const
        {
            int start = static_cast<int>(batch_idx) * batch_size_;
            int end = std::min(start + batch_size_, num_samples_);
            int actual_bs = end - start;

            Batch batch;
            batch.features.resize(actual_bs, num_features_);
            batch.labels.resize(actual_bs, num_labels_);

            for (int i = 0; i < actual_bs; ++i)
            {
                int idx = indices_[start + i];
                for (int f = 0; f < num_features_; ++f)
                    batch.features(i, f) = features_(idx, f);
                for (int l = 0; l < num_labels_; ++l)
                    batch.labels(i, l) = labels_(idx, l);
            }

            return batch;
        }

        // ---------------------------------------------------------------
        //  Iterator
        // ---------------------------------------------------------------

        DataLoader::Iterator::Iterator(DataLoader* loader, std::size_t batch_idx)
            : loader_(loader), batch_idx_(batch_idx) {}

        Batch DataLoader::Iterator::operator*() const
        {
            return loader_->get_batch(batch_idx_);
        }

        DataLoader::Iterator& DataLoader::Iterator::operator++()
        {
            ++batch_idx_;
            return *this;
        }

        bool DataLoader::Iterator::operator!=(const Iterator& other) const
        {
            return batch_idx_ != other.batch_idx_;
        }

        DataLoader::Iterator DataLoader::begin()
        {
            return Iterator(this, 0);
        }

        DataLoader::Iterator DataLoader::end()
        {
            return Iterator(this, num_batches_);
        }
    }
}

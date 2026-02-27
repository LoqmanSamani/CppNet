/**
 * @file tensor_io.cpp
 * @brief NumPy .npy format reader/writer for Eigen tensors
 *
 * .npy v1.0 layout:
 *   6 bytes  magic: \x93NUMPY
 *   1 byte   major version (1)
 *   1 byte   minor version (0)
 *   2 bytes  HEADER_LEN (little-endian uint16)
 *   HEADER_LEN bytes  Python dict string, padded with spaces to align to 64
 *   raw data (C-contiguous, little-endian float32)
 *
 * This implementation writes and reads little-endian float32 only.
 */

#include "CppNet/utils/tensor_io.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <vector>

namespace CppNet
{
    namespace Utils
    {

        static void write_npy_header(std::ofstream& ofs,
                                     const std::vector<int>& shape)
        {
            std::ostringstream dict;
            dict << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
            for (std::size_t i = 0; i < shape.size(); ++i)
            {
                dict << shape[i];
                if (i + 1 < shape.size()) dict << ", ";
            }
            if (shape.size() == 1) dict << ",";  // trailing comma for 1-tuple
            dict << "), }\n";

            std::string dict_str = dict.str();
            std::size_t preamble = 10;  // 6 magic + 1 major + 1 minor + 2 header_len
            std::size_t total = preamble + dict_str.size();
            std::size_t pad = (64 - (total % 64)) % 64;
            dict_str.insert(dict_str.size() - 1, pad, ' ');  // pad before \n

            uint16_t header_len = static_cast<uint16_t>(dict_str.size());

            const char magic[] = "\x93NUMPY";
            ofs.write(magic, 6);

            uint8_t major = 1, minor = 0;
            ofs.write(reinterpret_cast<const char*>(&major), 1);
            ofs.write(reinterpret_cast<const char*>(&minor), 1);
            ofs.write(reinterpret_cast<const char*>(&header_len), 2);
            ofs.write(dict_str.data(), static_cast<std::streamsize>(dict_str.size()));
        }

        static std::vector<int> read_npy_header(std::ifstream& ifs)
        {
            char magic[6];
            ifs.read(magic, 6);
            if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
                throw std::runtime_error("Not a valid .npy file (bad magic)");

            uint8_t major, minor;
            ifs.read(reinterpret_cast<char*>(&major), 1);
            ifs.read(reinterpret_cast<char*>(&minor), 1);

            uint16_t header_len;
            ifs.read(reinterpret_cast<char*>(&header_len), 2);

            std::string header(header_len, ' ');
            ifs.read(&header[0], header_len);

            if (header.find("<f4") == std::string::npos &&
                header.find("float32") == std::string::npos)
                throw std::runtime_error("Only float32 (<f4) .npy files are supported");

            auto pos_start = header.find("'shape': (");
            if (pos_start == std::string::npos)
                throw std::runtime_error("Malformed .npy header: missing shape");
            pos_start += 10;  // skip "'shape': ("
            auto pos_end = header.find(')', pos_start);

            std::string shape_str = header.substr(pos_start, pos_end - pos_start);
            std::vector<int> shape;
            std::istringstream ss(shape_str);
            std::string token;
            while (std::getline(ss, token, ','))
            {
                std::size_t s = token.find_first_not_of(" ");
                std::size_t e = token.find_last_not_of(" ");
                if (s == std::string::npos) continue;
                shape.push_back(std::stoi(token.substr(s, e - s + 1)));
            }

            return shape;
        }

        void save_npy(const std::string& path, const Eigen::Tensor<float, 1>& tensor)
        {
            std::ofstream ofs(path, std::ios::binary);
            if (!ofs) throw std::runtime_error("Cannot open " + path + " for writing");

            std::vector<int> shape = { static_cast<int>(tensor.dimension(0)) };
            write_npy_header(ofs, shape);

            int numel = shape[0];
            ofs.write(reinterpret_cast<const char*>(tensor.data()),
                      static_cast<std::streamsize>(numel * sizeof(float)));
        }

        void save_npy(const std::string& path, const Eigen::Tensor<float, 2>& tensor)
        {
            std::ofstream ofs(path, std::ios::binary);
            if (!ofs) throw std::runtime_error("Cannot open " + path + " for writing");

            std::vector<int> shape = {
                static_cast<int>(tensor.dimension(0)),
                static_cast<int>(tensor.dimension(1))
            };
            write_npy_header(ofs, shape);

            int numel = shape[0] * shape[1];
            ofs.write(reinterpret_cast<const char*>(tensor.data()),
                      static_cast<std::streamsize>(numel * sizeof(float)));
        }

        void save_npy(const std::string& path, const Eigen::Tensor<float, 3>& tensor)
        {
            std::ofstream ofs(path, std::ios::binary);
            if (!ofs) throw std::runtime_error("Cannot open " + path + " for writing");

            std::vector<int> shape = {
                static_cast<int>(tensor.dimension(0)),
                static_cast<int>(tensor.dimension(1)),
                static_cast<int>(tensor.dimension(2))
            };
            write_npy_header(ofs, shape);

            int numel = shape[0] * shape[1] * shape[2];
            ofs.write(reinterpret_cast<const char*>(tensor.data()),
                      static_cast<std::streamsize>(numel * sizeof(float)));
        }

        void save_npy(const std::string& path, const Eigen::Tensor<float, 4>& tensor)
        {
            std::ofstream ofs(path, std::ios::binary);
            if (!ofs) throw std::runtime_error("Cannot open " + path + " for writing");

            std::vector<int> shape = {
                static_cast<int>(tensor.dimension(0)),
                static_cast<int>(tensor.dimension(1)),
                static_cast<int>(tensor.dimension(2)),
                static_cast<int>(tensor.dimension(3))
            };
            write_npy_header(ofs, shape);

            int numel = shape[0] * shape[1] * shape[2] * shape[3];
            ofs.write(reinterpret_cast<const char*>(tensor.data()),
                      static_cast<std::streamsize>(numel * sizeof(float)));
        }

        Eigen::Tensor<float, 1> load_npy_1d(const std::string& path)
        {
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs) throw std::runtime_error("Cannot open " + path + " for reading");

            auto shape = read_npy_header(ifs);
            if (shape.size() != 1)
                throw std::runtime_error("Expected 1D tensor in " + path);

            Eigen::Tensor<float, 1> tensor(shape[0]);
            ifs.read(reinterpret_cast<char*>(tensor.data()),
                     static_cast<std::streamsize>(shape[0] * sizeof(float)));
            return tensor;
        }

        Eigen::Tensor<float, 2> load_npy_2d(const std::string& path)
        {
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs) throw std::runtime_error("Cannot open " + path + " for reading");

            auto shape = read_npy_header(ifs);
            if (shape.size() != 2)
                throw std::runtime_error("Expected 2D tensor in " + path);

            Eigen::Tensor<float, 2> tensor(shape[0], shape[1]);
            ifs.read(reinterpret_cast<char*>(tensor.data()),
                     static_cast<std::streamsize>(shape[0] * shape[1] * sizeof(float)));
            return tensor;
        }

        Eigen::Tensor<float, 3> load_npy_3d(const std::string& path)
        {
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs) throw std::runtime_error("Cannot open " + path + " for reading");

            auto shape = read_npy_header(ifs);
            if (shape.size() != 3)
                throw std::runtime_error("Expected 3D tensor in " + path);

            Eigen::Tensor<float, 3> tensor(shape[0], shape[1], shape[2]);
            ifs.read(reinterpret_cast<char*>(tensor.data()),
                     static_cast<std::streamsize>(shape[0] * shape[1] * shape[2] * sizeof(float)));
            return tensor;
        }

        Eigen::Tensor<float, 4> load_npy_4d(const std::string& path)
        {
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs) throw std::runtime_error("Cannot open " + path + " for reading");

            auto shape = read_npy_header(ifs);
            if (shape.size() != 4)
                throw std::runtime_error("Expected 4D tensor in " + path);

            Eigen::Tensor<float, 4> tensor(shape[0], shape[1], shape[2], shape[3]);
            ifs.read(reinterpret_cast<char*>(tensor.data()),
                     static_cast<std::streamsize>(
                         shape[0] * shape[1] * shape[2] * shape[3] * sizeof(float)));
            return tensor;
        }
    }
}

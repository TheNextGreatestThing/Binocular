#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <math.h>
#include <vector>
#include <functional>
#include <filesystem>
#include <random>
#include <cassert>
#include <memory>
#include <type_traits>
#include <execution>
#include <algorithm>
#include <variant>
#include <stdexcept>
#include <cstdarg>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef AC_WITH_CUDA
inline cudaError_t cudaCheckError(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(result) << std::endl;
        exit(result);
    }
    return result;
}


int createDirectory(const char *path) {
    struct stat st;

    if (stat(path, &st) == 0) {
        // The directory already exists.
        if (S_ISDIR(st.st_mode)) {
            return 0; // Directory exists, no need to create.
        } else {
            // A file with the same name exists, which prevents directory creation.
            return -1;
        }
    } else {
        // The directory doesn't exist, so create it.
        if (mkdir(path, 0777) == 0) {
            return 0; // Directory created successfully.
        } else {
            return -1; // Error in directory creation.
        }
    }
}

// Function to read a CUDA matrix from a file
void readCUDAMatrixFromFile(float* d_matrix, const char* filename, int numRows, int numCols) {
     std::ifstream inputFile(filename);
    if (!inputFile)
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<float> h_matrix(numRows * numCols);
    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numCols; ++j)
        {
            if (!(inputFile >> h_matrix[i * numCols + j]))
            {
                std::cerr << "Error reading file: " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    // Copy data from host to GPU
    cudaCheckError(cudaMemcpy(d_matrix, h_matrix.data(), numRows * numCols * sizeof(float), cudaMemcpyHostToDevice));

}

// Function to write a CUDA matrix to a file
void writeCUDAMatrixToFile(float* d_matrix, const char* filename, int numRows, int numCols) {
         std::vector<float> h_matrix(numRows * numCols);
    
    // Copy data from GPU to host
    cudaCheckError(cudaMemcpy(h_matrix.data(), d_matrix, numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost));

    std::ofstream outputFile(filename);
    if (!outputFile)
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numCols; ++j)
        {
            outputFile << h_matrix[i * numCols + j];
            if (j < numCols - 1)
                outputFile << ",";
        }
        outputFile << std::endl;
    }

}


// CUDA Kernels
__global__ void multiplyKernel(const float* A, int rows_A, int cols_A,
                               const float* B, int rows_B, int cols_B,
                               float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_A && col < cols_B) {
        float value = 0.0f;
        for (int i = 0; i < cols_A; ++i) {
            value += A[row * cols_A + i] * B[i * cols_B + col];
        }
        C[row * cols_B + col] = value;
    }
}

__global__ void multiplyScalarKernel(const float* A, int rows, int cols, float scalar, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] * scalar;
    }
}

__global__ void addScalarKernel(const float* A, int rows, int cols, float scalar, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] + scalar;
    }
}

__global__ void elementWiseMultiplyKernel(const float* A, const float* B, int rows, int cols, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] * B[row * cols + col];
    }
}

__global__ void elementWiseAddKernel(const float* A, const float* B, int rows, int cols, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
    }
}

__global__ void transposeKernel(const float* A, int rows, int cols, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < cols && col < rows) {
        C[row * rows + col] = A[col * cols + row];
    }
}
#endif

#ifdef AC_USE_CPU
#ifdef AC_WITH_EIGEN
    #include <Eigen/Dense>
#endif
#endif

#ifdef AC_USE_GPU

#ifdef AC_WITH_VIENNACL
#ifdef AC_VIENNACL_WITH_OPENCL
#ifndef VIENNACL_WITH_OPENCL
#define VIENNACL_WITH_OPENCL
#endif
#include <OpenCL/opencl.h>
#endif
 

#ifdef AC_VIENNACL_WITH_CUDA
#include <CudaMatrix.h>
#endif
    #include <viennacl/matrix.hpp>
    #include <viennacl/linalg/prod.hpp>
    #include <viennacl/ocl/context.hpp>
    #include <viennacl/linalg/norm_1.hpp>
    #include <viennacl/linalg/matrix_operations.hpp>
    #include <viennacl/linalg/fft_operations.hpp>
    #include <viennacl/ocl/backend.hpp>

#endif
#endif

namespace AlphaCore {
std::random_device rd;
std::mt19937 gen(rd());

using namespace std;
enum Arguments { GPUMATRIX_NO_INIT = -1, CPUMATRIX_NO_INIT = 0, GLOROT_INIT = 1, XAVIER_INIT = 1, HE_INIT = 2, RANDOM_INIT = 3 };

static float random(float min, float max, float factor) {
    std::uniform_int_distribution<> distr(min, max);

    return (distr(gen) * factor);
}

static float random_gaussian(float mean, float deviation, float factor) {
    std::normal_distribution<float> distr(mean, deviation);
    return (distr(gen) * factor);
}



#ifdef AC_USE_GPU

#ifdef AC_WITH_CUDA
class CudaMatrix {
public:
    float* data;
    int rows;
    int cols;

public:
    // Constructor
    CudaMatrix(int rows, int cols) : rows(rows), cols(cols) {
        cudaMallocManaged(&data, rows * cols * sizeof(float));
    }

    //Copy constructor
    CudaMatrix(const CudaMatrix& other) : rows(other.rows), cols(other.cols) {
        this->rows = other.rows;
       this->cols = other.cols;
      cudaMallocManaged(&data, rows * cols * sizeof(float));

        // Copy the data from the other matrix
        for (int i = 0; i < rows * cols; ++i) {
            data[i] = other.data[i];
        }
    }

    //Move constructor
     CudaMatrix(CudaMatrix&& other) noexcept : data(other.data), rows(other.rows), cols(other.cols) {
        // Set the source object to a valid but unspecified state
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
    }

    //Default constructor
    CudaMatrix() {
      data = nullptr;
      rows = GPUMATRIX_NO_INIT;
      cols = GPUMATRIX_NO_INIT;
    };

    // Destructor
    ~CudaMatrix() {
        cudaFree(data);
    }

    void unaryExpr(auto ufunc) {
      for (int rw = 0; rw < rows; rw++) {
        for (int cl = 0; cl < cols; cl++) {
          this->get(rw, cl) = ufunc(get(rw, cl));
        }
      }
    }

    void operator=(const CudaMatrix& other) {
      this->rows = other.rows;
       this->cols = other.cols;
      cudaMallocManaged(&data, rows * cols * sizeof(float));

        // Copy the data from the other matrix
        for (int i = 0; i < rows * cols; ++i) {
            data[i] = other.data[i];
        }
    };

    // Matrix-Matrix Multiplication
    CudaMatrix matrixProd(const CudaMatrix& other) const {
        if (cols != other.rows) {
            std::cerr << "Error: Matrix dimensions mismatch for multiplication.\n";
            exit(1);
        }

        CudaMatrix result(rows, other.cols);

        dim3 blockSize(16, 16);
        dim3 gridSize((result.cols + blockSize.x - 1) / blockSize.x, (result.rows + blockSize.y - 1) / blockSize.y);

        // Perform matrix multiplication on the GPU
        multiplyKernel<<<gridSize, blockSize>>>(data, rows, cols, other.data, other.rows, other.cols, result.data);
        cudaDeviceSynchronize();

        return result;
    }

    // Matrix-Float Multiplication
    CudaMatrix operator*(float scalar) const {
        CudaMatrix result(rows, cols);

        dim3 blockSize(16, 16);
        dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

        // Perform matrix-float multiplication on the GPU
        multiplyScalarKernel<<<gridSize, blockSize>>>(data, rows, cols, scalar, result.data);
        cudaDeviceSynchronize();

        return result;
    }

    void operator*=(float scalar) const {

        dim3 blockSize(16, 16);
        dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

        // Perform matrix-float multiplication on the GPU
        multiplyScalarKernel<<<gridSize, blockSize>>>(data, rows, cols, scalar, data);
        cudaDeviceSynchronize();
    }

    // Matrix-Float Addition
    CudaMatrix operator+(float scalar) const {
        CudaMatrix result(rows, cols);

        dim3 blockSize(16, 16);
        dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

        // Perform matrix-float addition on the GPU
        addScalarKernel<<<gridSize, blockSize>>>(data, rows, cols, scalar, result.data);
        cudaDeviceSynchronize();

        return result;
    }

    void operator+=(float scalar) const {

        dim3 blockSize(16, 16);
        dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

        // Perform matrix-float addition on the GPU
        addScalarKernel<<<gridSize, blockSize>>>(data, rows, cols, scalar, data);
        cudaDeviceSynchronize();
    }

    // Element-wise Matrix-Matrix Multiplication
    CudaMatrix elementProd(const CudaMatrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            std::cerr << "Error: Element-wise matrix dimensions mismatch for multiplication.\n";
            exit(1);
        }

        CudaMatrix result(rows, cols);

        dim3 blockSize(16, 16);
        dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

        // Perform element-wise matrix multiplication on the GPU
        elementWiseMultiplyKernel<<<gridSize, blockSize>>>(data, other.data, rows, cols, result.data);
        cudaDeviceSynchronize();

        return result;
    }

    void operator+=(CudaMatrix m) {
      dim3 blockSize(16, 16);
        dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
elementWiseAddKernel<<<gridSize, blockSize>>>(data, m.data, rows, cols, data);
        cudaDeviceSynchronize();
    }

    // Element-wise Matrix-Matrix Addition
    CudaMatrix operator+(const CudaMatrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            std::cerr << "Error: Element-wise matrix dimensions mismatch for addition.\n";
            exit(1);
        }

        CudaMatrix result(rows, cols);

        dim3 blockSize(16, 16);
        dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

        // Perform element-wise matrix addition on the GPU
        elementWiseAddKernel<<<gridSize, blockSize>>>(data, other.data, rows, cols, result.data);
        cudaDeviceSynchronize();

        return result;
    }

    // Matrix Transposition
    CudaMatrix transpose() const {
        CudaMatrix result(cols, rows);

        dim3 blockSize(16, 16);
        dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

        // Perform matrix transposition on the GPU
        transposeKernel<<<gridSize, blockSize>>>(data, rows, cols, result.data);
        cudaDeviceSynchronize();

        return result;
    }

    void transposeInPlace() {
        CudaMatrix result(cols, rows);

        dim3 blockSize(16, 16);
        dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

        // Perform matrix transposition on the GPU
        transposeKernel<<<gridSize, blockSize>>>(data, rows, cols, result.data);
        cudaDeviceSynchronize();

        // Swap dimensions and data with the result matrix
        std::swap(rows, result.rows);
        std::swap(cols, result.cols);
        std::swap(data, result.data);
    }

    //Copy from array
    void copyFromArray(const float* arrayPtr) {
        for (int i = 0; i < rows * cols; ++i) {
            data[i] = arrayPtr[i];
        }
    }

    float& get(int row, int col) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            std::cerr << "Error: Invalid row or column index.\n";
            exit(1);
        }
        return data[row * cols + col];
    }

    // Print the matrix
    void print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << data[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    void toFile(string fpath) {
      writeCUDAMatrixToFile(this->data, fpath.c_str(), rows, cols);
    }

    void fromFile(string fpath) {
      readCUDAMatrixFromFile(this->data, fpath.c_str(), rows, cols);
    }
};

////////////
class GPUMatrix {
public:
    int rows;
    int cols;
   GPUMatrix(GPUMatrix&& m) {
        this->rows = m.rows;
        this->cols = m.cols;
        this->data = std::move(m.data);
    }
   GPUMatrix(const GPUMatrix& m) {
        this->rows = m.rows;
        this->cols = m.cols;
        this->data = m.data;
    }

    void operator=(const GPUMatrix& m) {
        this->rows = m.rows;
        this->cols = m.cols;
        this->data = m.data;
    }
    
    GPUMatrix(int rows, int cols) : data(rows, cols) {
        this->rows = rows;
        this->cols = cols;
        data = CudaMatrix(this->rows, this->cols);
    }
    
    CudaMatrix data;
    
    GPUMatrix() {
        this->rows = GPUMATRIX_NO_INIT;
        this->cols = GPUMATRIX_NO_INIT;
        data = CudaMatrix(GPUMATRIX_NO_INIT, GPUMATRIX_NO_INIT);
    };
    
    void copy(GPUMatrix&& m) {
        rows = m.rows;
        cols = m.cols;
        data = m.data;
    }
    
    
    static GPUMatrix fromArray(std::vector<float>& arr) {
        GPUMatrix m(arr.size(), 1);
        m.data.copyFromArray(arr.data());
        return m;
    }
    
    void toFile(std::string name) {
       writeCUDAMatrixToFile(data.data, name.c_str(), rows, cols);
    }

    static GPUMatrix subtract(GPUMatrix&& m1, GPUMatrix&& m2) {
        m1.scale(-1.0f);
        return GPUMatrix::add(std::move(m1), std::move(m2));
    }

    GPUMatrix(string filepath, int r, int c) : rows(r), cols(c){
            data = MatfromFile(filepath, r, c);
        }
    
    template<class t>
    void add(const t& n) {
      #ifdef AC_WITH_CUDA
      (this->data) += n.data;
      #endif
      #ifdef AC_WITH_VIENNACL
      (this->data) += n;
      #endif
    }
    
    float firstval() {
        return data.get(0, 0);
    }
    
    void scale(float n) {
        (this->data) *= n;
    }
    
    void add(float n) {
        (this->data) += n;
    }
    
   static GPUMatrix add(GPUMatrix&& a, GPUMatrix&& b) {
        GPUMatrix result(a.rows, a.cols);
        result.data = a.data + b.data;
        return result;
      }
    
    std::vector<float> toArray() {
        std::vector<float> arr;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                arr.push_back(data.get(i, j));
            }
        }
        return arr;
    }
    
    void pmap(std::function<float(float)> func) {
      #ifdef AC_WITH_CUDA
        data.unaryExpr(func);
      #endif
      #ifdef AC_WITH_VIENNACL
        data = MatrixXf(data.unaryExpr(func));
      #endif
      
    }
    
    template<typename... Args, typename Func>
    void tmap(Func func, Args&&... args) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data.get(i,j) = func(data.get(i,j), std::move(args)...);
            }
        }

    }
    
    template<typename... Args, typename Func>
    void tnmap(Func func, Args&&... args) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data.get(i,j) = func(data.get(i,j), i, std::move(args)...);
            }
        }

    }
    
    static GPUMatrix multiply(GPUMatrix&& a, GPUMatrix&& b) {
        GPUMatrix result(a.rows, a.cols);
        result.data = a.data.matrixProd(b.data);
        return result;
    }
    
    static GPUMatrix map(GPUMatrix&& matrix, float (*func)(float)) {
        GPUMatrix result(matrix.rows, matrix.cols);
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.cols; j++) {
                float val = matrix.data.get(i,j);
                result.data.get(i,j) = func(val);
            }
        }
        return result;
    }

    template<typename... Args, typename Func>
    static GPUMatrix tmap(GPUMatrix&& matrix, Func func, Args&&... args) {
        GPUMatrix newm = matrix;
        for (int i = 0; i < newm.rows; i++) {
            for (int j = 0; j < newm.cols; j++) {
                float val = newm.data.get(i,j);
                newm.data.get(i,j) = func(val, std::move(args)...);
            }
        }
        return newm;
    }
    
    template<typename... Args, typename Func>
    static GPUMatrix tnmap(GPUMatrix&& matrix, Func func, Args&&... args) {
        GPUMatrix newm = matrix;
        for (int i = 0; i < newm.rows; i++) {
            for (int j = 0; j < newm.cols; j++) {
                float val = newm.data.get(i,j);
                newm.data.get(i,j) = func(val, i, std::move(args)...);
            }
        }
        return newm;
    }
    
    static std::vector<float> toArray(GPUMatrix&& m) {
        std::vector<float> arr;
        arr.reserve(m.rows * m.cols);
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                arr.push_back(m.data.get(i,j));
            }
        }
        return arr;
    }
    
     static GPUMatrix matrixProduct(GPUMatrix&& vect1, GPUMatrix&& vect2) {
        assert(vect1.cols == vect2.rows);
        GPUMatrix result(vect1.rows, vect2.cols);
        result.data =  vect1.data.matrixProd(vect2.data);
        return result;
    }
    
    static GPUMatrix matrixProductTransRight(GPUMatrix&& vect1, GPUMatrix&& vect2) {
        return matrixProduct(move(vect1), GPUMatrix::transpose(move(vect2)));
    }
    static GPUMatrix matrixProductTransLeft(GPUMatrix&& vect1, GPUMatrix&& vect2) {
        return matrixProduct(GPUMatrix::transpose(move(vect1)), move(vect2));
    }
    
    void print() {
        data.print();
    }
    
     void transpose() {
        this->data.transposeInPlace();
         auto oldrows = this->rows;
         this->rows = this->cols;
         this->cols = oldrows;
    }
    
    static GPUMatrix transpose(GPUMatrix&& matrix) {
        GPUMatrix result = GPUMatrix(matrix.cols, matrix.rows);
        result.data = matrix.data;
        result.data.transposeInPlace();
        return result;
    }

    static void MattoFile(CudaMatrix m, string fpath) {
      m.toFile(fpath.c_str());
    }

    static CudaMatrix MatfromFile(string fpath, int rows, int cols) {
      CudaMatrix m(rows, cols);
      m.fromFile(fpath);
      return m;
    }
};

#endif

#ifdef AC_WITH_VIENNACL

typedef viennacl::matrix<float> matrixf;
#ifdef VIENNACL_WITH_OPENCL
void initGpuAcceleration() {
    std::vector<cl_device_id> device_id_array;

    //Get all available devices
    viennacl::ocl::platform pf;
    std::cout << "Platform info: " << pf.info() << std::endl;
    std::vector<viennacl::ocl::device> devices = pf.devices(CL_DEVICE_TYPE_DEFAULT);
    std::cout << devices[0].name() << std::endl;
    std::cout << "Number of devices for custom context: " << devices.size() << std::endl;

    //Set up context using all found devices:
    for (std::size_t i=0; i<devices.size(); ++i)
    {
        device_id_array.push_back(devices[i].id());
    }

    std::cout << "Creating context..." << std::endl;
    cl_int err;
    cl_context my_context = clCreateContext(0, cl_uint(device_id_array.size()), &(device_id_array[0]), NULL, NULL, &err);
    VIENNACL_ERR_CHECK(err);
    std::cout << "Context created" << std::endl;
}

std::string gpuSource() {
    return viennacl::ocl::current_context().current_device().name();
}
#endif

//GPU ViennaCL Matrix class
class GPUMatrix {
    
public:
    template <typename NumericT>
    static viennacl::matrix<NumericT> MatfromFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
            
        // Read matrix dimensions
        std::size_t num_rows, num_cols;
        file.read(reinterpret_cast<char*>(&num_rows), sizeof(std::size_t));
        file.read(reinterpret_cast<char*>(&num_cols), sizeof(std::size_t));
        
        // Create a new matrix with the read dimensions
        viennacl::matrix<NumericT> matrix(num_rows, num_cols);
        
        // Read matrix data
        file.read(reinterpret_cast<char*>(matrix.handle().ram_handle().get()), matrix.internal_size() * sizeof(NumericT));
        
        file.close();
        
        return matrix;
    }
    
    template <typename NumericT>
    static void MattoFile(const viennacl::matrix<NumericT>& matrix, const std::string& filename) {
        std::ofstream file(filename, std::ios::binary | std::ios::trunc);
            
            // Check if the file is opened successfully
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open the file for writing.");
            }
            
            // Write matrix dimensions
            std::size_t num_rows = matrix.size1();
            std::size_t num_cols = matrix.size2();
            file.write(reinterpret_cast<const char*>(&num_rows), sizeof(std::size_t));
            file.write(reinterpret_cast<const char*>(&num_cols), sizeof(std::size_t));
            
            // Write matrix data
            file.write(reinterpret_cast<const char*>(matrix.handle().ram_handle().get()), matrix.internal_size() * sizeof(NumericT));
            
            file.close();
    }
    

    int rows;
    int cols;
    GPUMatrix(GPUMatrix&& m) {
        this->rows = m.rows;
        this->cols = m.cols;
        this->data = std::move(m.data);
    }
    GPUMatrix(const GPUMatrix& m) {
        this->rows = m.rows;
        this->cols = m.cols;
        this->data = m.data;
    }
    
    GPUMatrix(const matrixf& d) {
        this->rows = d.size1();
        this->cols = d.size2();
        this->data = d;
    }
    
    GPUMatrix(int rows, int cols) : data(rows, cols) {
        this->rows = rows;
        this->cols = cols;
    }
    
    GPUMatrix() {
        this->rows = GPUMATRIX_NO_INIT;
        this->cols = GPUMATRIX_NO_INIT;
        this->data(0, 0);
    }

    matrixf data;
    
        GPUMatrix(string filepath, int r, int c) : rows(r), cols(c){
            data = MatfromFile<float>(filepath);
        }

    void copy(GPUMatrix&& m) {
        rows = m.rows;
        cols = m.cols;
        data.resize(rows, cols);
        data = m.data;
    }
    
    
    static GPUMatrix fromArray(std::vector<float>& arr) {
        GPUMatrix m(arr.size(), 1);
        for (int i = 0; i < arr.size(); i++) {
            m.data(i, 0) = arr[i];
        }
        return m;
    }
    

    void add(GPUMatrix n) {
        data = data + n.data;
    }
    
    float firstval() {
        return data(0, 0);
    }
    
    void scale(viennacl::scalar<float> n) {
        data = data * n;
    }

    
    static GPUMatrix subtract(GPUMatrix&& a, GPUMatrix&& b) {
        GPUMatrix result(a.rows, a.cols);
        result.data = matrixf(a.data - b.data);
        return result;
    }
    
    std::vector<float> toArray() {
        std::vector<float> arr;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                arr.push_back(data(i, j));
            }
        }
        return arr;
    }
    
    void pmap(std::function<float(float)> func) {
        for(int i = 0; i < data.size2(); i++) {
            for(int j = 0; j < data.size1(); j++) {
                data(j,i) = func(data(j,i));
            }
        }
    }
    
    template<typename... Args, typename Func>
    void tmap(Func func, Args&&... args) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data(i,j) = func(data(i,j), std::move(args)...);
            }
        }

    }
    
    template<typename... Args, typename Func>
    void tnmap(Func func, Args&&... args) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data(i,j) = func(data(i,j), i, std::move(args)...);
            }
        }

    }
    
    static GPUMatrix multiply(GPUMatrix&& a, GPUMatrix&& b) {
        GPUMatrix result(a.rows, a.cols);
        result.data = viennacl::linalg::element_prod(a.data, b.data);
        return result;
    }
    
    static GPUMatrix divide(GPUMatrix&& a, GPUMatrix&& b) {
        GPUMatrix result(a.rows, a.cols);
        result.data =  viennacl::linalg::element_div(a.data, b.data);
        return result;
    }

    static GPUMatrix divide(GPUMatrix&& a, float b) {
        GPUMatrix result(a.rows, a.cols);
        result.data = a.data * b;
        return result;
    }
    
    static GPUMatrix map(GPUMatrix&& matrix, float (*func)(float)) {
        GPUMatrix result(matrix.rows, matrix.cols);
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.cols; j++) {
                float val = matrix.data(i,j);
                result.data(i,j) = func(val);
            }
        }
        return result;
    }

    template<typename... Args, typename Func>
    static GPUMatrix tmap(GPUMatrix&& matrix, Func func, Args&&... args) {
        GPUMatrix newm = matrix;
        for (int i = 0; i < newm.rows; i++) {
            for (int j = 0; j < newm.cols; j++) {
                float val = newm.data(i,j);
                newm.data(i,j) = func(val, std::move(args)...);
            }
        }
        return newm;
    }
    
    template<typename... Args, typename Func>
    static GPUMatrix tnmap(GPUMatrix&& matrix, Func func, Args&&... args) {
        GPUMatrix newm = matrix;
        for (int i = 0; i < newm.rows; i++) {
            for (int j = 0; j < newm.cols; j++) {
                float val = newm.data(i,j);
                newm.data(i,j) = func(val, i, std::move(args)...);
            }
        }
        return newm;
    }
    
    static std::vector<float> toArray(GPUMatrix&& m) {
        std::vector<float> arr;
        arr.reserve(m.rows * m.cols);
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                arr.push_back(m.data(i,j));
            }
        }
        return arr;
    }
    
    static GPUMatrix matrixProduct(GPUMatrix&& vect1, GPUMatrix&& vect2) {
        assert(vect1.cols == vect2.rows);
        GPUMatrix result(vect1.rows, vect2.cols);
        result.data = viennacl::linalg::prod(vect1.data, vect2.data);
        return result;
    }
    
    static GPUMatrix matrixProductTransRight(GPUMatrix&& vect1, GPUMatrix&& vect2) {
        GPUMatrix result(viennacl::linalg::prod(vect1.data, viennacl::trans(vect2.data)));
        return result;
    }
    static GPUMatrix matrixProductTransLeft(GPUMatrix&& vect1, GPUMatrix&& vect2) {
        GPUMatrix result(viennacl::linalg::prod(viennacl::trans(vect1.data), vect2.data));
        return result;
    }
    
    void print() {
        std::cout << data;
    }
};

#endif

class GPUFeedForwardNeuralNetwork;


static float sigmoid(float x, int node, GPUMatrix&& qweights, GPUMatrix&& layernodes) {
    return 1 / (1 + exp(-x));
}

static float sigmoidder(float x,  int node, GPUMatrix&& qweights, GPUMatrix&& layernodes) {
    return x * (1 - x);
}

static float relu(float x, int node, GPUMatrix&& qweights, GPUMatrix&& layernodes) {
    if (x>0){
        return x;
    } else {
        return 0;
    }
}

static float reluder(float x,  int node, GPUMatrix&& qweights, GPUMatrix&& layernodes) {
    if (x>=0){
        return 1.f;
    } else {
        return 0.f;
    }
}

static float lrelu(float x, int node, GPUMatrix&& qweights, GPUMatrix&& layernodes) {
    if (x>=0){
        return x;
    } else {
        return 0.01f*x;
    }

}

static float eluder(float x, int node, GPUMatrix&& qweights, GPUMatrix&& layernodes) {
    if (x>=0){
        return 1.f;
    } else {
        return exp(x);
    }
}

static float elu(float x, int node, GPUMatrix&& qweights, GPUMatrix&& layernodes) {
    if (x>=0){
        return x;
    } else {
        return std::exp(x-1.f);
    }

}

static float lreluder(float x, int node, GPUMatrix&& qweights, GPUMatrix&& layernodes) {
    if (x>0){
        return 1.f;
    } else {
        return 0.01f;
    }
}

static float tanhf(float x, int node, GPUMatrix&& qweights, GPUMatrix&& layernodes) {
    return std::tanh(x);
}


static float tanhfder(float x, int node, GPUMatrix&& qweights, GPUMatrix&& layernodes) {
    return 1.0f - (x * x);
}

#ifdef AC_WITH_CUDA
static float softmaxder(float x, int node, GPUMatrix&& qweights, GPUMatrix&& layernodes) {
    std::vector<float> outputvect;
    int j = node;
    for (int i = 0; i < layernodes.data.rows * layernodes.data.cols; i++) {
        if (i == j) {
            outputvect.push_back(x * (1.f-x));
        } else {
            float iv = layernodes.data.get(i,0);
            float jv = layernodes.data.get(j,0);
            outputvect.push_back(-1.0f * iv * jv);
        }
    }
    return outputvect[j];
}
#endif

#ifdef AC_WITH_VIENNACL
static float softmaxder(float x, int node, GPUMatrix&& qweights, GPUMatrix&& layernodes) {
    std::vector<float> outputvect;
    int j = node;
    for (int i = 0; i < layernodes.data.size1() * layernodes.data.size2(); i++) {
        if (i == j) {
            outputvect.push_back(x * (1.f-x));
        } else {
            float iv = layernodes.data(i,0);
            float jv = layernodes.data(j,0);
            outputvect.push_back(-1.0f * iv * jv);
        }
    }
    return outputvect[j];
}
#endif

static float softmax(float x, int node, GPUMatrix&& qweights, GPUMatrix&& layernodes) {
    std::vector<float> nodes = GPUMatrix::toArray(std::move(layernodes));
    std::vector<float> raised;
    std::vector<float> finals;
    float sum = 0;
    for (auto n : nodes) {
        float num = exp(n);
        sum += num;
        raised.push_back(num);
    }
    for (auto rn : raised) {
        finals.push_back(rn/sum);
    }
    return finals[node];
}

//GPU Feedforward Neural Network Class Implementation
class GPUFeedForwardNeuralNetwork {
private:
    static float NaNchecker(float x) {
        assert(isnan(x)==false);
        return x;
    }
    
    bool ncheck = true;
    
public:
    
    void doNanCheck(bool n) {
        ncheck = n;
    }

public:
    int inputNodes;
    int outputNodes;
    int hiddenLayers;
    int networkSize;
    float learningRate;
    std::vector<float(*)(float, int, GPUMatrix&&, GPUMatrix&&)> activations;
    std::vector<float(*)(float, int, GPUMatrix&&, GPUMatrix&&)> derivatives;
    vector<vector<int>> elementwiseSkipConnections;
public:
    std::vector<GPUMatrix> weights;
    std::vector<GPUMatrix> biases;

    GPUFeedForwardNeuralNetwork(std::vector<int>&& structure, Arguments wInit) {
        for (int i = 0; i < structure.size()-1; i++) {
            activations.push_back(sigmoid);
            derivatives.push_back(sigmoidder);
        }
        inputNodes = structure[0];
        outputNodes = structure[structure.size() - 1];
        hiddenLayers = structure.size() - 2;
        networkSize = structure.size();
        learningRate = 0.1;
        
            for (int i = 1; i < structure.size(); i++) {
                GPUMatrix weight(structure[i], structure[i - 1]);
                if (wInit == 1) {
                weight.pmap([&](float x) {
                    float fanin = structure[i-1];
                    float fanout = structure[i];
                    float variance = 2.0 / (fanin + fanout);
                    float standarddeviation = sqrt(variance);
                    return (float)random_gaussian(0.0f, standarddeviation, 1.0f);
                });
                } else if (wInit == 2) {
                    weight.pmap([&](float x) {
                    float fanin = structure[i-1];
                    float standarddeviation = sqrt(2.0/fanin);
                    return random_gaussian(0.0f, standarddeviation, 1.0f);
                    });
                } else if (wInit == 3) {
                    weight.pmap([&](float x) {
                        return random(-1000000, 1000000, 0.000001);
                    });
                } else {
                    std::cout << "Invalid Weight Initialization" << std::endl;
                    assert(false);
                }
                weights.push_back(weight);
                GPUMatrix bias(structure[i], 1);
                #ifdef AC_WITH_CUDA
                bias.data = CudaMatrix(structure[i], 1);
                #endif
                #ifdef AC_WITH_VIENNACL
                bias.data = matrixf(structure[i], 1);
                #endif
                
                biases.push_back(bias);
                
            }
        
    }

 
    
    GPUFeedForwardNeuralNetwork(vector<int>&& structure, string filepath) {
        for (int i = 0; i < structure.size()-1; i++) {
            activations.push_back(sigmoid);
            derivatives.push_back(sigmoidder);
        }
        inputNodes = structure[0];
        outputNodes = structure[structure.size() - 1];
        hiddenLayers = structure.size() - 2;
        networkSize = structure.size();
        learningRate = 0.1;

        for (int i = 1; i < structure.size(); i++) {
            string wPath = filepath;
            wPath += "/";
            wPath += "w";
            wPath += to_string(i-1);
            wPath += ".csv";
            GPUMatrix weight(wPath, structure[i], structure[i-1]);
            weights.push_back(weight);
            string bPath = filepath;
            bPath += "/";
            bPath += "b";
            bPath += to_string(i-1);
            bPath += ".csv";
            GPUMatrix bias(bPath, structure[i], 1);
            biases.push_back(bias);

        }
    }

    void save(string filepath) {
        createDirectory(filepath.c_str());
        for (int i = 0; i < weights.size(); i++) {
            #ifdef AC_WITH_CUDA
            string newPath = filepath;
            newPath += "/";
            newPath += "w";
            newPath += to_string(i);
            newPath += ".csv";
            GPUMatrix::MattoFile(weights[i].data, newPath);
            #else
            string newPath = filepath;
            newPath += "/";
            newPath += "w";
            newPath += to_string(i);
            newPath += ".csv";
            GPUMatrix::MattoFile<float>(weights[i].data, newPath);
            #endif
        }
        for (int i = 0; i < biases.size(); i++) {
            #ifdef AC_WITH_CUDA
            string newPath = filepath;
            newPath += "/";
            newPath += "b";
            newPath += to_string(i);
            newPath += ".csv";
            GPUMatrix::MattoFile(biases[i].data, newPath);
            #else
            string newPath = filepath;
            newPath += "/";
            newPath += "b";
            newPath += to_string(i);
            newPath += ".csv";
            GPUMatrix::MattoFile<float>(biases[i].data, newPath);
            #endif
        }
    }

    std::vector<float> feedforward(std::vector<float>&& input_array) {
        GPUMatrix inputs = (GPUMatrix&&)GPUMatrix::fromArray(input_array);
        //NAN CHECK
        if (ncheck) {
        inputs.tmap(NaNchecker);
        }
        
        vector<GPUMatrix> skipLayers;
        skipLayers.reserve(networkSize);
        GPUMatrix currentLayer = (GPUMatrix&&)GPUMatrix::matrixProduct((GPUMatrix&&)weights[0], (GPUMatrix&&)inputs);
        currentLayer.add(biases[0]);
        currentLayer.tnmap(activations[0], move(weights[0]), move(currentLayer));
        for (int j = 0; j < networkSize; j++) {
            skipLayers.push_back(GPUMatrix());
        }
        for (int j = 0; j < elementwiseSkipConnections.size(); j++) {
            auto& connection = elementwiseSkipConnections[j];
            if (connection[0] == 0) {
                skipLayers[connection[0]].copy(move(inputs));
            } else if (connection[0] == 1) {
                skipLayers[connection[0]].copy(move(currentLayer));
            }
        }
        for (int i = 1; i < networkSize - 1; i++) {
            currentLayer.copy((GPUMatrix&&)GPUMatrix::matrixProduct(move(weights[i]), move(currentLayer)));
            currentLayer.add(biases[i]);
                currentLayer.tnmap(activations[i], move(weights[i]), move(currentLayer));
            for (int j = 0; j < elementwiseSkipConnections.size(); j++) {
                auto& connection = elementwiseSkipConnections[j];
                if (connection[0] == i+1) {
                    if (skipLayers[connection[0]].rows == GPUMATRIX_NO_INIT) {
                        skipLayers[connection[0]].copy(move(currentLayer));
                    } else {
                        skipLayers[connection[0]].add(move(currentLayer));
                    }
                }
                if (connection[1] == i+1) {
                    currentLayer.add(skipLayers[connection[0]]);
                }
            }
        }
        return currentLayer.toArray();
    }

    std::vector<GPUMatrix> debugfeedforward(std::vector<float>&& input_array) {
        GPUMatrix inputs = (GPUMatrix&&)GPUMatrix::fromArray(input_array);
        
        //NAN CHECK
        if (ncheck) {
        inputs.tmap(NaNchecker);
        }
        vector<GPUMatrix> skipLayers;
        skipLayers.reserve(networkSize);
    
        vector<GPUMatrix> pLayers;
        pLayers.push_back(inputs);
        GPUMatrix currentLayer = (GPUMatrix&&)GPUMatrix::matrixProduct(move(weights[0]), move(inputs));
        currentLayer.add(biases[0]);
        currentLayer.tnmap(activations[0], move(weights[0]), move(currentLayer));
        pLayers.push_back(currentLayer);
        for (int j = 0; j < networkSize; j++) {
            skipLayers.push_back(GPUMatrix());
        }
        for (int j = 0; j < elementwiseSkipConnections.size(); j++) {
            auto& connection = elementwiseSkipConnections[j];
            if (connection[0] == 0) {
                skipLayers[connection[0]].copy(move(inputs));
            } else if (connection[0] == 1) {
                skipLayers[connection[0]].copy(move(currentLayer));
            }
        }
        for (int i = 1; i < networkSize - 1; i++) {
            currentLayer.copy((GPUMatrix&&)GPUMatrix::matrixProduct(move(weights[i]), move(currentLayer)));
            currentLayer.add(biases[i]);
            currentLayer.tnmap(activations[i], move(weights[i]), move(currentLayer));
            for (int j = 0; j < elementwiseSkipConnections.size(); j++) {
                auto& connection = elementwiseSkipConnections[j];
                if (connection[0] == i+1) {
                    if (skipLayers[connection[0]].rows == GPUMATRIX_NO_INIT) {
                        skipLayers[connection[0]].copy(move(currentLayer));
                    } else {
                        skipLayers[connection[0]].add(move(currentLayer));
                    }
                }
                if (connection[1] == i+1) {
                    currentLayer.add(skipLayers[connection[0]]);
                }            }
            pLayers.push_back(currentLayer);
        }
        return pLayers;
    }
    
    void backpropagate(std::vector<float>&& input_array, std::vector<float>&& target_array) {
        GPUMatrix inputs = (GPUMatrix&&)GPUMatrix::fromArray(input_array);
        GPUMatrix targets = (GPUMatrix&&)GPUMatrix::fromArray(target_array);
        
        //NAN CHECK
        if (ncheck) {
        inputs.pmap(NaNchecker);
        targets.pmap(NaNchecker);
        }
        
        vector<GPUMatrix> skipLayers;
        skipLayers.reserve(networkSize);
    
        vector<GPUMatrix> pLayers;
        pLayers.push_back(inputs);
        GPUMatrix currentLayer = (GPUMatrix&&)GPUMatrix::matrixProduct(move(weights[0]), move(inputs));
        currentLayer.add(biases[0]);
        currentLayer.tnmap(activations[0], move(weights[0]), move(currentLayer));
        pLayers.push_back(currentLayer);
        for (int j = 0; j < networkSize; j++) {
            skipLayers.push_back(GPUMatrix());
        }
        for (int j = 0; j < elementwiseSkipConnections.size(); j++) {
            auto& connection = elementwiseSkipConnections[j];
            if (connection[0] == 0) {
                skipLayers[connection[0]].copy(move(inputs));
            } else if (connection[0] == 1) {
                skipLayers[connection[0]].copy(move(currentLayer));
            }
        }
        for (int i = 1; i < networkSize - 1; i++) {
            currentLayer.copy((GPUMatrix&&)GPUMatrix::matrixProduct(move(weights[i]), move(currentLayer)));
            currentLayer.add(biases[i]);
            currentLayer.tnmap(activations[i], move(weights[i]), move(currentLayer));
            for (int j = 0; j < elementwiseSkipConnections.size(); j++) {
                auto& connection = elementwiseSkipConnections[j];
                if (connection[0] == i+1) {
                    if (skipLayers[connection[0]].rows == GPUMATRIX_NO_INIT) {
                        skipLayers[connection[0]].copy(move(currentLayer));
                    } else {
                        skipLayers[connection[0]].add(move(currentLayer));
                    }
                }
                if (connection[1] == i+1) {
                    currentLayer.add(skipLayers[connection[0]]);
                }
            }
            pLayers.push_back(currentLayer);
        }
        vector<GPUMatrix> networkGradients(networkSize, GPUMatrix());
        GPUMatrix currentErrors = (GPUMatrix&&)GPUMatrix::subtract(move(targets), move(currentLayer));
        for (int i = networkSize - 1; i > 0; i--) {
            GPUMatrix currentGradients = (GPUMatrix&&)GPUMatrix::tnmap(move(currentLayer), derivatives[i-1], move(weights[i-1]), move(currentLayer));
            currentGradients.copy((GPUMatrix&&)GPUMatrix::multiply(move(currentGradients), move(currentErrors)));
            currentGradients.scale(learningRate);
            biases[i - 1].add(currentGradients);
            currentGradients.copy((GPUMatrix&&)GPUMatrix::matrixProductTransRight(move(currentGradients), move(pLayers[i - 1])));
            for (int j = 0; j > elementwiseSkipConnections.size(); j++) {
                auto& connection = elementwiseSkipConnections[j];
                if (connection[1] == i) {
                    if (networkGradients[i].rows == GPUMATRIX_NO_INIT) {
                        networkGradients[i].copy(move(currentGradients));
                    } else {
                        networkGradients[i].add(move(currentGradients));
                    }
                }
                if (connection[0] == i) {
                    currentGradients.add(networkGradients[connection[1]]);
                }
            }
            weights[i - 1].add(currentGradients);
            
            //NAN CHECK
            if (ncheck) {
            weights[i-1].pmap(NaNchecker);
            biases[i-1].pmap(NaNchecker);
            }
            //
            if (i != 1) {
                currentErrors.copy((GPUMatrix&&)GPUMatrix::matrixProductTransLeft(move(weights[i-1]), move(currentErrors)));
            //NAN CHECK
                if (ncheck) {
                currentErrors.pmap(NaNchecker);
                }
            //
            
                currentLayer.copy(move(pLayers[i - 1]));
            }
        }
    }

    GPUMatrix debugbackpropagate(std::vector<float>&& input_array,std::vector<float>&& target_array) {
        GPUMatrix inputs = (GPUMatrix&&)GPUMatrix::fromArray(input_array);
        GPUMatrix targets = (GPUMatrix&&)GPUMatrix::fromArray(target_array);
        
        //NAN CHECK
        if (ncheck) {
        inputs.pmap(NaNchecker);
        targets.pmap(NaNchecker);
        }
        
        vector<GPUMatrix> skipLayers;
        skipLayers.reserve(networkSize);
    
        vector<GPUMatrix> pLayers;
        pLayers.push_back(inputs);
        GPUMatrix currentLayer = (GPUMatrix&&)GPUMatrix::matrixProduct(move(weights[0]), move(inputs));
        currentLayer.add(biases[0]);
        currentLayer.tnmap(activations[0], move(weights[0]), move(currentLayer));
        pLayers.push_back(currentLayer);
        for (int j = 0; j < networkSize; j++) {
            skipLayers.push_back(GPUMatrix());
        }
        for (int j = 0; j < elementwiseSkipConnections.size(); j++) {
            auto& connection = elementwiseSkipConnections[j];
            if (connection[0] == 0) {
                skipLayers[connection[0]].copy(move(inputs));
            } else if (connection[0] == 1) {
                skipLayers[connection[0]].copy(move(currentLayer));
            }
        }
        for (int i = 1; i < networkSize - 1; i++) {
            currentLayer.copy((GPUMatrix&&)GPUMatrix::matrixProduct(move(weights[i]), move(currentLayer)));
            currentLayer.add(biases[i]);
            currentLayer.tnmap(activations[i], move(weights[i]), move(currentLayer));
            for (int j = 0; j < elementwiseSkipConnections.size(); j++) {
                auto& connection = elementwiseSkipConnections[j];
                if (connection[0] == i+1) {
                    if (skipLayers[connection[0]].rows == GPUMATRIX_NO_INIT) {
                        skipLayers[connection[0]].copy(move(currentLayer));
                    } else {
                        skipLayers[connection[0]].add(move(currentLayer));
                    }
                }
                if (connection[1] == i+1) {
                    currentLayer.add(skipLayers[connection[0]]);
                }
            }
            pLayers.push_back(currentLayer);
        }
        vector<GPUMatrix> networkGradients(networkSize, GPUMatrix());
        GPUMatrix currentErrors = (GPUMatrix&&)GPUMatrix::subtract(move(targets), move(currentLayer));
        GPUMatrix losserrors = currentErrors;
        for (int i = networkSize - 1; i > 0; i--) {
            GPUMatrix currentGradients = (GPUMatrix&&)GPUMatrix::tnmap(move(currentLayer), derivatives[i-1], move(weights[i-1]), move(currentLayer));
            currentGradients.copy((GPUMatrix&&)GPUMatrix::multiply(move(currentGradients), move(currentErrors)));
            currentGradients.scale(learningRate);
            biases[i - 1].add(currentGradients);
            currentGradients.copy((GPUMatrix&&)GPUMatrix::matrixProductTransRight(move(currentGradients), move(pLayers[i - 1])));
            for (int j = 0; j > elementwiseSkipConnections.size(); j++) {
                auto& connection = elementwiseSkipConnections[j];
                if (connection[1] == i) {
                    if (networkGradients[i].rows == GPUMATRIX_NO_INIT) {
                        networkGradients[i].copy(move(currentGradients));
                    } else {
                        networkGradients[i].add(move(currentGradients));
                    }
                }
                if (connection[0] == i) {
                    currentGradients.add(networkGradients[connection[1]]);
                }
            }
            weights[i - 1].add(currentGradients);
            
            //NAN CHECK
            if (ncheck) {
            weights[i-1].pmap(NaNchecker);
            biases[i-1].pmap(NaNchecker);
            }
            //
            if (i != 1) {
                currentErrors.copy((GPUMatrix&&)GPUMatrix::matrixProductTransLeft(move(weights[i-1]), move(currentErrors)));
            //NAN CHECK
                if (ncheck) {
                currentErrors.pmap(NaNchecker);
                }
            //
            
                currentLayer.copy(move(pLayers[i - 1]));
            }
        }
        return losserrors;
    }
    void addElementwiseSkipConnection(int a, int b) {
        vector<int> connection = {a,b};
        assert(connection[1]-connection[0] > 1);
        assert(connection[0] < connection[1]);
        elementwiseSkipConnections.push_back(connection);
    }
    
    void trainset(std::vector<std::vector<float>> inputset, std::vector<std::vector<float>> targetset, int iterations) {
        assert(inputset.size() == targetset.size());
        assert(inputset[0].size() == targetset[0].size());
        assert(inputset[0].size() == inputNodes);
        for (int j = 0; j < iterations; j++)
        for (int i = 0; i < inputset.size(); i++) {
            this->backpropagate(std::move(inputset[i]),std::move(targetset[i]));
        }
        
    }
    
    void setLearningRate(float x) {
        learningRate = x;
    }
    
    void setActivations(int layer, float(*func)(float, int, GPUMatrix&&, GPUMatrix&&)) {
        activations[layer] = func;
    }

    void setDerivatives(int layer, float(*func)(float, int, GPUMatrix&&, GPUMatrix&&)) {
        derivatives[layer] = func;
    }

    void gaussianVary(float deviation, float variationrate, float varitationStrength) {
        auto Mutate = [&](float val, float r) {
            if (AlphaCore::random(0, 100000, 0.00001) < r) {
                return val +( varitationStrength * random_gaussian(0.0f, deviation, 1.0f));
            }
            else {
                return val;
            }
        };
        for (auto& w : weights) {
            w.tmap(Mutate, variationrate);
        }

        for (auto& b : biases) {
            b.tmap(Mutate, variationrate);
        }
    }
    
    void vary(float rate, float variationStrength) {
        auto Mutate = [&](float val, float r) {
            if (AlphaCore::random(0, 100000, 0.00001) < r) {
                return variationStrength * AlphaCore::random(-100000, 100000, 0.000001);
            }
            else {
                return val;
            }
        };
        for (auto& w : weights) {
            w.tmap(Mutate, rate);
        }

        for (auto& b : biases) {
            b.tmap(Mutate, rate);
        }
    }
};


#endif

#ifdef AC_USE_CPU
#ifdef AC_WITH_EIGEN
using namespace Eigen;
using Eigen::MatrixXf;
const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");


//CPU Matrix class that uses SIMD acceleration
class CPUMatrix {
public:
    static Eigen::MatrixXf fromFile(std::string file, int rs, int cs) {

      std::ifstream in(file);
      
      std::string line;

      int row = 0;
      int col = 0;

      Eigen::MatrixXf res = Eigen::MatrixXf(rs, cs);

      if (in.is_open()) {

        while (std::getline(in, line)) {

          char *ptr = (char *) line.c_str();
          int len = line.length();

          col = 0;

          char *start = ptr;
          for (int i = 0; i < len; i++) {

            if (ptr[i] == ',') {
              res(row, col++) = atof(start);
              start = ptr + i + 1;
            }
          }
          res(row, col) = atof(start);

          row++;
        }

        in.close();
      }
      return res;
    }
    
    int rows;
    int cols;
   CPUMatrix(CPUMatrix&& m) {
        this->rows = m.rows;
        this->cols = m.cols;
        this->data = std::move(m.data);
    }
   CPUMatrix(const CPUMatrix& m) {
        this->rows = m.rows;
        this->cols = m.cols;
        this->data = m.data;
    }
    
    CPUMatrix(int rows, int cols) : data(rows, cols) {
        this->rows = rows;
        this->cols = cols;
        data = Eigen::MatrixXf::Zero(this->rows, this->cols);
    }
    
    Eigen::MatrixXf data;
    
    CPUMatrix(std::string filepath, int r, int c) : rows(r), cols(c){
        data = fromFile(filepath, r, c);
    }
    
    CPUMatrix() {
        this->rows = CPUMATRIX_NO_INIT;
        this->cols = CPUMATRIX_NO_INIT;
        data = Eigen::MatrixXf(0, 0);
    };
    
    void copy(CPUMatrix&& m) {
        rows = m.rows;
        cols = m.cols;
        data = m.data;
    }
    
    
    static CPUMatrix fromArray(std::vector<float>& arr) {
        CPUMatrix m(arr.size(), 1);
        m.data = Eigen::Map<Eigen::MatrixXf>(arr.data(), arr.size(), 1);
        return m;
    }
    
    void toFile(std::string name) {
        std::ofstream file(name.c_str());
        file << data.format(CSVFormat);
    }
    
    void randomize(float factor = 1) {
        data = MatrixXf::Random(rows, cols).array()*factor;
    }
    
    template<class t>
    void add(const t& n) {
        if constexpr (std::is_same<t, CPUMatrix>::value) {
            data.array() += n.data.array();
        } else {
            data.array() += n;
        }
    }
    
    float firstval() {
        return data(0, 0);
    }
    
    void scale(float n) {
        data.array() *= n;
    }
    
    void add(float n) {
        data.array() += n;
    }
    
    static CPUMatrix subtract(CPUMatrix&& a, CPUMatrix&& b) {
        CPUMatrix result(a.rows, a.cols);
        result.data = MatrixXf((a.data).array() - (b.data).array());
        return result;
    }
    static CPUMatrix add(CPUMatrix&& a, CPUMatrix&& b) {
        CPUMatrix result(a.rows, a.cols);
        result.data = MatrixXf((a.data).array() + (b.data).array());
        return result;
    }
    
    
    std::vector<float> toArray() {
        std::vector<float> arr;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                arr.push_back(data(i, j));
            }
        }
        return arr;
    }
    
    void pmap(std::function<float(float)> func) {

        data = MatrixXf(data.unaryExpr(func));

    }
    
    template<typename... Args, typename Func>
    void tmap(Func func, Args&&... args) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data(i,j) = func(data(i,j), std::move(args)...);
            }
        }

    }
    
    template<typename... Args, typename Func>
    void tnmap(Func func, Args&&... args) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data(i,j) = func(data(i,j), i, std::move(args)...);
            }
        }

    }
    
    static CPUMatrix multiply(CPUMatrix&& a, CPUMatrix&& b) {
        CPUMatrix result(a.rows, a.cols);
        result.data = MatrixXf((a.data).array() * (b.data).array());
        return result;
    }
    
    static CPUMatrix divide(CPUMatrix&& a, CPUMatrix&& b) {
        CPUMatrix result(a.rows, a.cols);
        result.data =  MatrixXf((a.data).array() / (b.data).array());
        return result;
    }

    static CPUMatrix divide(CPUMatrix&& a, float b) {
        CPUMatrix result(a.rows, a.cols);
        result.data = MatrixXf((a.data).array() * b);
        return result;
    }
    
    static CPUMatrix map(CPUMatrix&& matrix, float (*func)(float)) {
        CPUMatrix result(matrix.rows, matrix.cols);
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.cols; j++) {
                float val = matrix.data(i,j);
                result.data(i,j) = func(val);
            }
        }
        return result;
    }

    template<typename... Args, typename Func>
    static CPUMatrix tmap(CPUMatrix&& matrix, Func func, Args&&... args) {
        CPUMatrix newm = matrix;
        for (int i = 0; i < newm.rows; i++) {
            for (int j = 0; j < newm.cols; j++) {
                float val = newm.data(i,j);
                newm.data(i,j) = func(val, std::move(args)...);
            }
        }
        return newm;
    }
    
    template<typename... Args, typename Func>
    static CPUMatrix tnmap(CPUMatrix&& matrix, Func func, Args&&... args) {
        CPUMatrix newm = matrix;
        for (int i = 0; i < newm.rows; i++) {
            for (int j = 0; j < newm.cols; j++) {
                float val = newm.data(i,j);
                newm.data(i,j) = func(val, i, std::move(args)...);
            }
        }
        return newm;
    }
    
    static std::vector<float> toArray(CPUMatrix&& m) {
        std::vector<float> arr;
        arr.reserve(m.rows * m.cols);
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                arr.push_back(m.data(i,j));
            }
        }
        return arr;
    }
    
    static CPUMatrix matrixProduct(CPUMatrix&& vect1, CPUMatrix&& vect2) {
        assert(vect1.cols == vect2.rows);
        CPUMatrix result(vect1.rows, vect2.cols);
        result.data =  vect1.data * vect2.data;
        return result;
    }
    
    static CPUMatrix matrixProductTransRight(CPUMatrix&& vect1, CPUMatrix&& vect2) {
        return matrixProduct(move(vect1), CPUMatrix::transpose(move(vect2)));
    }
    static CPUMatrix matrixProductTransLeft(CPUMatrix&& vect1, CPUMatrix&& vect2) {
        return matrixProduct(CPUMatrix::transpose(move(vect1)), move(vect2));
    }
    
    void print() {
        std::cout << data;
    }
    
     void transpose() {
        this->data.transposeInPlace();
         auto oldrows = this->rows;
         this->rows = this->cols;
         this->cols = oldrows;
    }
    
    static CPUMatrix transpose(CPUMatrix&& matrix) {
        CPUMatrix result = CPUMatrix(matrix.cols, matrix.rows);
        result.data = matrix.data;
        result.data.transposeInPlace();
        return result;
    }
};

#endif

template <typename R, typename ... Types>
static constexpr size_t getArgumentCount( R(*f)(Types ...))
{
   return sizeof...(Types);
}


static float sigmoid(float x, int node, CPUMatrix&& qweights, CPUMatrix&& layernodes) {
    return 1 / (1 + exp(-x));
}

static float sigmoidder(float x,  int node, CPUMatrix&& qweights, CPUMatrix&& layernodes) {
    return x * (1 - x);
}

static float relu(float x, int node, CPUMatrix&& qweights, CPUMatrix&& layernodes) {
    if (x>0){
        return x;
    } else {
        return 0;
    }
}

static float reluder(float x,  int node, CPUMatrix&& qweights, CPUMatrix&& layernodes) {
    if (x>=0){
        return 1.f;
    } else {
        return 0.f;
    }
}

static float lrelu(float x, int node, CPUMatrix&& qweights, CPUMatrix&& layernodes) {
    if (x>=0){
        return x;
    } else {
        return 0.01f*x;
    }

}

static float eluder(float x, int node, CPUMatrix&& qweights, CPUMatrix&& layernodes) {
    if (x>=0){
        return 1.f;
    } else {
        return exp(x);
    }
}

static float elu(float x, int node, CPUMatrix&& qweights, CPUMatrix&& layernodes) {
    if (x>=0){
        return x;
    } else {
        return std::exp(x-1.f);
    }

}

static float lreluder(float x, int node, CPUMatrix&& qweights, CPUMatrix&& layernodes) {
    if (x>0){
        return 1.f;
    } else {
        return 0.01f;
    }
}

static float tanhf(float x, int node, CPUMatrix&& qweights, CPUMatrix&& layernodes) {
    return std::tanhf(x);
}


static float tanhfder(float x, int node, CPUMatrix&& qweights, CPUMatrix&& layernodes) {
    return 1.0f - (x * x);
}

static float softmaxder(float x, int node, CPUMatrix&& qweights, CPUMatrix&& layernodes) {
    std::vector<float> outputvect;
    int j = node;
    for (int i = 0; i < layernodes.data.size(); i++) {
        if (i == j) {
            outputvect.push_back(x * (1.f-x));
        } else {
            float iv = layernodes.data(i,0);
            float jv = layernodes.data(j,0);
            outputvect.push_back(-1.0f * iv * jv);
        }
    }
    return outputvect[j];
}

static float softmax(float x, int node, CPUMatrix&& qweights, CPUMatrix&& layernodes) {
    std::vector<float> nodes = CPUMatrix::toArray(std::move(layernodes));
    std::vector<float> raised;
    std::vector<float> finals;
    float sum = 0;
    for (auto n : nodes) {
        float num = exp(n);
        sum += num;
        raised.push_back(num);
    }
    for (auto rn : raised) {
        finals.push_back(rn/sum);
    }
    return finals[node];
}

class CPUFeedForwardNeuralNetwork {
private:
    static float NaNchecker(float x) {
        assert(isnan(x)==false);
        return x;
    }
    
    bool ncheck = true;
public:
    
    void doNanCheck(bool n) {
        ncheck = n;
    }

    int inputNodes;
    int outputNodes;
    int hiddenLayers;
    int networkSize;
    float learningRate;
    vector<float(*)(float, int, CPUMatrix&&, CPUMatrix&&)> activations;
    vector<float(*)(float, int, CPUMatrix&&, CPUMatrix&&)> derivatives;
    vector<vector<int>> elementwiseSkipConnections;
    
    vector<CPUMatrix> weights;
    vector<CPUMatrix> biases;

    CPUFeedForwardNeuralNetwork(vector<int>&& structure, Arguments wInit) {
        for (int i = 0; i < structure.size()-1; i++) {
            activations.push_back(sigmoid);
            derivatives.push_back(sigmoidder);
        }
        inputNodes = structure[0];
        outputNodes = structure[structure.size() - 1];
        hiddenLayers = structure.size() - 2;
        networkSize = structure.size();
        learningRate = 0.1;
        
            for (int i = 1; i < structure.size(); i++) {
                CPUMatrix weight(structure[i], structure[i - 1]);
                if (wInit == 1) {
                weight.pmap([&](float x) {
                    float fanin = structure[i-1];
                    float fanout = structure[i];
                    float variance = 2.0 / (fanin + fanout);
                    float standarddeviation = sqrt(variance);
                    return (float)random_gaussian(0.0f, standarddeviation, 1.0f);
                });
                } else if (wInit == 2) {
                    weight.pmap([&](float x) {
                    float fanin = structure[i-1];
                    float standarddeviation = sqrt(2.0/fanin);
                    return random_gaussian(0.0f, standarddeviation, 1.0f);
                    });
                } else if (wInit == 3) {
                    weight.pmap([&](float x) {
                        return random(-1000000, 1000000, 0.000001);
                    });
                } else {
                    cout << "Invalid Weight Initialization" << endl;
                    assert(false);
                }
                weights.push_back(weight);
                CPUMatrix bias(structure[i], 1);
                bias.data = MatrixXf(MatrixXf::Zero(structure[i], 1));
                biases.push_back(bias);
                
            }
        
    }
    
    CPUFeedForwardNeuralNetwork(vector<int>&& structure, string filepath) {
        for (int i = 0; i < structure.size()-1; i++) {
            activations.push_back(sigmoid);
            derivatives.push_back(sigmoidder);
        }
        inputNodes = structure[0];
        outputNodes = structure[structure.size() - 1];
        hiddenLayers = structure.size() - 2;
        networkSize = structure.size();
        learningRate = 0.1;
        
        for (int i = 1; i < structure.size(); i++) {
            string wPath = filepath;
            wPath += "/";
            wPath += "w";
            wPath += to_string(i-1);
            wPath += ".csv";
            CPUMatrix weight(wPath, structure[i], structure[i-1]);
            weights.push_back(weight);
            string bPath = filepath;
            bPath += "/";
            bPath += "b";
            bPath += to_string(i-1);
            bPath += ".csv";
            CPUMatrix bias(bPath, structure[i], 1);
            biases.push_back(bias);
            
        }
    }
    
    void save(string filepath) {
        filesystem::create_directory(filepath);
        for (int i = 0; i < weights.size(); i++) {
            string newPath = filepath;
            newPath += "/";
            newPath += "w";
            newPath += to_string(i);
            newPath += ".csv";
            weights[i].toFile(newPath);
        }
        for (int i = 0; i < biases.size(); i++) {
            string newPath = filepath;
            newPath += "/";
            newPath += "b";
            newPath += to_string(i);
            newPath += ".csv";
            biases[i].toFile(newPath);
        }
    }

    vector<float> feedforward(vector<float>&& input_array) {
        CPUMatrix inputs = (CPUMatrix&&)CPUMatrix::fromArray(input_array);
        //NAN CHECK
        if (ncheck) {
        inputs.tmap(NaNchecker);
        }
        
        vector<CPUMatrix> skipLayers;
        skipLayers.reserve(networkSize);
        CPUMatrix currentLayer = (CPUMatrix&&)CPUMatrix::matrixProduct((CPUMatrix&&)weights[0], (CPUMatrix&&)inputs);
        currentLayer.add(biases[0]);
        currentLayer.tnmap(activations[0], move(weights[0]), move(currentLayer));
        for (int j = 0; j < networkSize; j++) {
            skipLayers.push_back(CPUMatrix());
        }
        for (int j = 0; j < elementwiseSkipConnections.size(); j++) {
            auto& connection = elementwiseSkipConnections[j];
            if (connection[0] == 0) {
                skipLayers[connection[0]].copy(move(inputs));
            } else if (connection[0] == 1) {
                skipLayers[connection[0]].copy(move(currentLayer));
            }
        }
        for (int i = 1; i < networkSize - 1; i++) {
            currentLayer.copy((CPUMatrix&&)CPUMatrix::matrixProduct(move(weights[i]), move(currentLayer)));
            currentLayer.add(biases[i]);
            currentLayer.tnmap(activations[i], move(weights[i]), move(currentLayer));
            for (int j = 0; j < elementwiseSkipConnections.size(); j++) {
                auto& connection = elementwiseSkipConnections[j];
                if (connection[0] == i+1) {
                    if (skipLayers[connection[0]].rows == CPUMATRIX_NO_INIT) {
                        skipLayers[connection[0]].copy(move(currentLayer));
                    } else {
                        skipLayers[connection[0]].add(move(currentLayer));
                    }
                }
                if (connection[1] == i+1) {
                    currentLayer.add(skipLayers[connection[0]]);
                }
            }
        }
        return currentLayer.toArray();
    }

    vector<CPUMatrix> debugfeedforward(vector<float>&& input_array) {
        CPUMatrix inputs = (CPUMatrix&&)CPUMatrix::fromArray(input_array);
        
        //NAN CHECK
        if (ncheck) {
        inputs.tmap(NaNchecker);
        }
        vector<CPUMatrix> skipLayers;
        skipLayers.reserve(networkSize);
    
        vector<CPUMatrix> pLayers;
        pLayers.push_back(inputs);
        CPUMatrix currentLayer = (CPUMatrix&&)CPUMatrix::matrixProduct(move(weights[0]), move(inputs));
        currentLayer.add(biases[0]);
        currentLayer.tnmap(activations[0], move(weights[0]), move(currentLayer));
        pLayers.push_back(currentLayer);
        for (int j = 0; j < networkSize; j++) {
            skipLayers.push_back(CPUMatrix());
        }
        for (int j = 0; j < elementwiseSkipConnections.size(); j++) {
            auto& connection = elementwiseSkipConnections[j];
            if (connection[0] == 0) {
                skipLayers[connection[0]].copy(move(inputs));
            } else if (connection[0] == 1) {
                skipLayers[connection[0]].copy(move(currentLayer));
            }
        }
        for (int i = 1; i < networkSize - 1; i++) {
            currentLayer.copy((CPUMatrix&&)CPUMatrix::matrixProduct(move(weights[i]), move(currentLayer)));
            currentLayer.add(biases[i]);
            currentLayer.tnmap(activations[i], move(weights[i]), move(currentLayer));
            for (int j = 0; j < elementwiseSkipConnections.size(); j++) {
                auto& connection = elementwiseSkipConnections[j];
                if (connection[0] == i+1) {
                    if (skipLayers[connection[0]].rows == CPUMATRIX_NO_INIT) {
                        skipLayers[connection[0]].copy(move(currentLayer));
                    } else {
                        skipLayers[connection[0]].add(move(currentLayer));
                    }
                }
                if (connection[1] == i+1) {
                    currentLayer.add(skipLayers[connection[0]]);
                }
            }
            pLayers.push_back(currentLayer);
        }
        return pLayers;
    }
    
    void backpropagate(vector<float>&& input_array, vector<float>&& target_array) {
        CPUMatrix inputs = (CPUMatrix&&)CPUMatrix::fromArray(input_array);
        CPUMatrix targets = (CPUMatrix&&)CPUMatrix::fromArray(target_array);
        
        //NAN CHECK
        if (ncheck) {
        inputs.pmap(NaNchecker);
        targets.pmap(NaNchecker);
        }
        
        vector<CPUMatrix> skipLayers;
        skipLayers.reserve(networkSize);
    
        vector<CPUMatrix> pLayers;
        pLayers.push_back(inputs);
        CPUMatrix currentLayer = (CPUMatrix&&)CPUMatrix::matrixProduct(move(weights[0]), move(inputs));
        currentLayer.add(biases[0]);
        currentLayer.tnmap(activations[0], move(weights[0]), move(currentLayer));
        pLayers.push_back(currentLayer);
        for (int j = 0; j < networkSize; j++) {
            skipLayers.push_back(CPUMatrix());
        }
        for (int j = 0; j < elementwiseSkipConnections.size(); j++) {
            auto& connection = elementwiseSkipConnections[j];
            if (connection[0] == 0) {
                skipLayers[connection[0]].copy(move(inputs));
            } else if (connection[0] == 1) {
                skipLayers[connection[0]].copy(move(currentLayer));
            }
        }
        for (int i = 1; i < networkSize - 1; i++) {
            currentLayer.copy((CPUMatrix&&)CPUMatrix::matrixProduct(move(weights[i]), move(currentLayer)));
            currentLayer.add(biases[i]);
            currentLayer.tnmap(activations[i], move(weights[i]), move(currentLayer));
            for (int j = 0; j < elementwiseSkipConnections.size(); j++) {
                auto& connection = elementwiseSkipConnections[j];
                if (connection[0] == i+1) {
                    if (skipLayers[connection[0]].rows == CPUMATRIX_NO_INIT) {
                        skipLayers[connection[0]].copy(move(currentLayer));
                    } else {
                        skipLayers[connection[0]].add(move(currentLayer));
                    }
                }
                if (connection[1] == i+1) {
                    currentLayer.add(skipLayers[connection[0]]);
                }
            }
            pLayers.push_back(currentLayer);
        }
        vector<CPUMatrix> networkGradients(networkSize, CPUMatrix());
        CPUMatrix currentErrors = (CPUMatrix&&)CPUMatrix::subtract(move(targets), move(currentLayer));
        for (int i = networkSize - 1; i > 0; i--) {
            CPUMatrix currentGradients = (CPUMatrix&&)CPUMatrix::tnmap(move(currentLayer), derivatives[i-1], move(weights[i-1]), move(currentLayer));
            currentGradients.copy((CPUMatrix&&)CPUMatrix::multiply(move(currentGradients), move(currentErrors)));
            currentGradients.scale(learningRate);
            biases[i - 1].add(currentGradients);
            currentGradients.copy((CPUMatrix&&)CPUMatrix::matrixProductTransRight(move(currentGradients), move(pLayers[i - 1])));
            for (int j = 0; j > elementwiseSkipConnections.size(); j++) {
                auto& connection = elementwiseSkipConnections[j];
                if (connection[1] == i) {
                    if (networkGradients[i].rows == CPUMATRIX_NO_INIT) {
                        networkGradients[i].copy(move(currentGradients));
                    } else {
                        networkGradients[i].add(move(currentGradients));
                    }
                }
                if (connection[0] == i) {
                    currentGradients.add(networkGradients[connection[1]]);
                }
            }
            weights[i - 1].add(currentGradients);
            
            //NAN CHECK
            if (ncheck) {
            weights[i-1].pmap(NaNchecker);
            biases[i-1].pmap(NaNchecker);
            }
            //
            if (i != 1) {
                currentErrors.copy((CPUMatrix&&)CPUMatrix::matrixProductTransLeft(move(weights[i-1]), move(currentErrors)));
            //NAN CHECK
                if (ncheck) {
                currentErrors.pmap(NaNchecker);
                }
            //
            
                currentLayer.copy(move(pLayers[i - 1]));
            }
        }
         
    }

    CPUMatrix debugbackpropagate(vector<float>&& input_array,vector<float>&& target_array) {
        CPUMatrix inputs = (CPUMatrix&&)CPUMatrix::fromArray(input_array);
        CPUMatrix targets = (CPUMatrix&&)CPUMatrix::fromArray(target_array);
        
        //NAN CHECK
        if (ncheck) {
        inputs.pmap(NaNchecker);
        targets.pmap(NaNchecker);
        }
        
        vector<CPUMatrix> skipLayers;
        skipLayers.reserve(networkSize);
    
        vector<CPUMatrix> pLayers;
        pLayers.push_back(inputs);
        CPUMatrix currentLayer = (CPUMatrix&&)CPUMatrix::matrixProduct(move(weights[0]), move(inputs));
        currentLayer.add(biases[0]);
        currentLayer.tnmap(activations[0], move(weights[0]), move(currentLayer));
        pLayers.push_back(currentLayer);
        for (int j = 0; j < networkSize; j++) {
            skipLayers.push_back(CPUMatrix());
        }
        for (int j = 0; j < elementwiseSkipConnections.size(); j++) {
            auto& connection = elementwiseSkipConnections[j];
            if (connection[0] == 0) {
                skipLayers[connection[0]].copy(move(inputs));
            } else if (connection[0] == 1) {
                skipLayers[connection[0]].copy(move(currentLayer));
            }
        }
        for (int i = 1; i < networkSize - 1; i++) {
            currentLayer.copy((CPUMatrix&&)CPUMatrix::matrixProduct(move(weights[i]), move(currentLayer)));
            currentLayer.add(biases[i]);
            currentLayer.tnmap(activations[i], move(weights[i]), move(currentLayer));
            for (int j = 0; j < elementwiseSkipConnections.size(); j++) {
                auto& connection = elementwiseSkipConnections[j];
                if (connection[0] == i+1) {
                    if (skipLayers[connection[0]].rows == CPUMATRIX_NO_INIT) {
                        skipLayers[connection[0]].copy(move(currentLayer));
                    } else {
                        skipLayers[connection[0]].add(move(currentLayer));
                    }
                }
                if (connection[1] == i+1) {
                    currentLayer.add(skipLayers[connection[0]]);
                }
            }
            pLayers.push_back(currentLayer);
        }
        vector<CPUMatrix> networkGradients(networkSize, CPUMatrix());
        CPUMatrix currentErrors = (CPUMatrix&&)CPUMatrix::subtract(move(targets), move(currentLayer));
        CPUMatrix losserrors = currentErrors;
        for (int i = networkSize - 1; i > 0; i--) {
            CPUMatrix currentGradients = (CPUMatrix&&)CPUMatrix::tnmap(move(currentLayer), derivatives[i-1], move(weights[i-1]), move(currentLayer));
            currentGradients.copy((CPUMatrix&&)CPUMatrix::multiply(move(currentGradients), move(currentErrors)));
            currentGradients.scale(learningRate);
            biases[i - 1].add(currentGradients);
            currentGradients.copy((CPUMatrix&&)CPUMatrix::matrixProductTransRight(move(currentGradients), move(pLayers[i - 1])));
            for (int j = 0; j > elementwiseSkipConnections.size(); j++) {
                auto& connection = elementwiseSkipConnections[j];
                if (connection[1] == i) {
                    if (networkGradients[i].rows == CPUMATRIX_NO_INIT) {
                        networkGradients[i].copy(move(currentGradients));
                    } else {
                        networkGradients[i].add(move(currentGradients));
                    }
                }
                if (connection[0] == i) {
                    currentGradients.add(networkGradients[connection[1]]);
                }
            }
            weights[i - 1].add(currentGradients);
            
            //NAN CHECK
            if (ncheck) {
            weights[i-1].pmap(NaNchecker);
            biases[i-1].pmap(NaNchecker);
            }
            //
            if (i != 1) {
                currentErrors.copy((CPUMatrix&&)CPUMatrix::matrixProductTransLeft(move(weights[i-1]), move(currentErrors)));
            //NAN CHECK
                if (ncheck) {
                currentErrors.pmap(NaNchecker);
                }
            //
            
                currentLayer.copy(move(pLayers[i - 1]));
            }
        }
         
        return losserrors;
    }

    
    void addElementwiseSkipConnection(int a, int b) {
        vector<int> connection = {a,b};
        assert(connection[1]-connection[0] > 1);
        assert(connection[0] < connection[1]);
        elementwiseSkipConnections.push_back(connection);
    }
    
    void trainset(vector<vector<float>> inputset, vector<vector<float>> targetset, int iterations) {
        assert(inputset.size() == targetset.size());
        assert(inputset[0].size() == targetset[0].size());
        assert(inputset[0].size() == inputNodes);
        for (int j = 0; j < iterations; j++)
        for (int i = 0; i < inputset.size(); i++) {
            this->backpropagate(move(inputset[i]), move(targetset[i]));
        }
        
    }
    
    void setLearningRate(float x) {
        learningRate = x;
    }
    
    void setActivations(int layer, float(*func)(float, int, CPUMatrix&&, CPUMatrix&&)) {
        activations[layer] = func;
    }

    void setDerivatives(int layer, float(*func)(float, int, CPUMatrix&&, CPUMatrix&&)) {
        derivatives[layer] = func;
    }

    void gaussianVary(float deviation, float variationrate, float varitationStrength) {
        auto Mutate = [&](float val, float r) {
            if (AlphaCore::random(0, 100000, 0.00001) < r) {
                return val +( varitationStrength * random_gaussian(0.0f, deviation, 1.0f));
            }
            else {
                return val;
            }
        };
        for (auto& w : weights) {
            w.tmap(Mutate, variationrate);
        }

        for (auto& b : biases) {
            b.tmap(Mutate, variationrate);
        }
    }
    
    void vary(float rate, float variationStrength) {
        auto Mutate = [&](float val, float r) {
            if (AlphaCore::random(0, 100000, 0.00001) < r) {
                return variationStrength * AlphaCore::random(-100000, 100000, 0.000001);
            }
            else {
                return val;
            }
        };
        for (auto& w : weights) {
            w.tmap(Mutate, rate);
        }

        for (auto& b : biases) {
            b.tmap(Mutate, rate);
        }
    }
};
#endif


}

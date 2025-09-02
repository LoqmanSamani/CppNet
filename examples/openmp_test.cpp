#include <iostream>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        std::cout << "Hello from thread " << omp_get_thread_num() 
                  << " out of " << omp_get_num_threads() << std::endl;
    }
    return 0;
}

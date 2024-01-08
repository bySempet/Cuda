#include <cuda_runtime.h>

void obtenerDatosGPU(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
        printf("Device name: %s\n", deviceProp.name);
        printf("Number of CUDA cores: %d\n", deviceProp.multiProcessorCount * _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
        printf("Clock rate: %d\n", deviceProp.clockRate);
        printf("Device memory: %zu\n", deviceProp.totalGlobalMem);
        printf("Memory Clock Rate (KHz): %d\n", deviceProp.memoryClockRate);
        printf("Memory Bus Width (bits): %d\n", deviceProp.memoryBusWidth);
        printf("Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth/8)/1.0e6);
    }

}
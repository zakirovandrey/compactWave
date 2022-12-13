#pragma once

#include <cuda_runtime.h>
#include "cuda_math.h"
#include <assert.h>
#include <iostream>

#ifdef USE_FLOAT
#define FTYPESIZE 4
typedef float ftype;
typedef float2 ftype2;
typedef float3 ftype3;
typedef float4 ftype4;
template<typename T1,typename T2>                         __host__ __device__ ftype2 make_ftype2(const T1& f1, const T2& f2) { return make_float2(f1,f2); }
template<typename T1,typename T2,typename T3>             __host__ __device__ ftype3 make_ftype3(const T1& f1, const T2& f2, const T3& f3) { return make_float3(f1,f2,f3); }
template<typename T1,typename T2,typename T3,typename T4> __host__ __device__ ftype4 make_ftype4(const T1& f1, const T2& f2, const T3& f3, const T4& f4) { return make_float4(f1,f2,f3,f4); }
template<typename T> __host__ __device__ ftype3 make_ftype3(const T& f) { return make_float3(f); }
typedef float fptype;
typedef float2 fptype2;
typedef float4 fptype4;
template<typename T1,typename T2> __host__ __device__ fptype2 make_fptype2(const T1& f1, const T2& f2) { return make_float2(f1,f2); }
template<typename T> __host__ __device__ float2 fp22ftype2(const T& f) { return f; }
template<typename T> __host__ __device__ float ftype2fptype(const T& f) { return f; }
template<typename T> __host__ __device__ float fptype2ftype(const T& f) { return f; }
#elif defined USE_DOUBLE
#include "cuda_math_double.h"
#define FTYPESIZE 8
typedef double ftype;
typedef double2 ftype2;
typedef double3 ftype3;
typedef double4 ftype4;
template<typename T1,typename T2>                         __host__ __device__ ftype2 make_ftype2(const T1& f1, const T2& f2) { return make_double2(f1,f2); }
template<typename T1,typename T2,typename T3>             __host__ __device__ ftype3 make_ftype3(const T1& f1, const T2& f2, const T3& f3) { return make_double3(f1,f2,f3); }
template<typename T1,typename T2,typename T3,typename T4> __host__ __device__ ftype4 make_ftype4(const T1& f1, const T2& f2, const T3& f3, const T4& f4) { return make_double4(f1,f2,f3,f4); }
template<typename T> __host__ __device__ ftype3 make_ftype3(const T& f) { return make_double3(f); }
typedef double fptype;
typedef double2 fptype2;
typedef double4 fptype4;
template<typename T1,typename T2> __host__ __device__ fptype2 make_fptype2(const T1& f1, const T2& f2) { return make_double2(f1,f2); }
template<typename T> __host__ __device__ double2 fp22ftype2(const T& f) { return f; }
template<typename T> __host__ __device__ double ftype2fptype(const T& f) { return f; }
template<typename T> __host__ __device__ double fptype2ftype(const T& f) { return f; }
#elif defined USE_HALF
typedef half fptype;
typedef half2 fptype2;
typedef float2 fptype4;
template<typename T1,typename T2> __host__ __device__ fptype2 make_fptype2(const T1& f1, const T2& f2) { return __float22half2_rn(make_float2(f1,f2)); }
template<typename T> __host__ __device__ float2 fp22ftype2(const T& f) { return __half22float2(f); }
template<typename T> __host__ __device__ half ftype2fptype(const T& f) { return __float2half(f); }
template<typename T> __host__ __device__ float fptype2ftype(const T& f) { return __half2float(f); }
#endif


#ifndef CHECK_ERROR_H
#define CHECK_ERROR_H
#define PRINT_LAST_ERROR() PrintLastError(__FILE__,__LINE__)
void PrintLastError(const char *file, int line);
#define CHECK_ERROR(err) CheckError( err, __FILE__,__LINE__)
bool CheckError( cudaError_t err, const char *file, int line);
#define CHECK_ERROR_DEVICE(err) CheckErrorDevice( err, __FILE__,__LINE__)
bool __device__ CheckErrorDevice( cudaError_t err, const char *file, int line);
void deviceDiagnostics();
#endif//CHECK_ERROR_H
#include <cuda.h>
#include <stdio.h>
#include "err.h"
namespace errors{
__managed__ cudaError_t last_err=cudaSuccess;
};

void PrintLastError(const char *file, int line) {
  cudaError_t err=cudaGetLastError();
  if(err!=cudaSuccess) fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
}
bool CheckError(cudaError_t err, const char *file, int line) {
  cudaError_t dev_err; cudaMemcpy(&dev_err, &errors::last_err, sizeof(cudaError_t), cudaMemcpyDefault);
  if(err==cudaSuccess && dev_err==cudaSuccess) return false;
  if(err==cudaSuccess) err = errors::last_err;
  fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
  return true;
}
bool __device__ CheckErrorDevice(cudaError_t err, const char *file, int line) {
  if(err==cudaSuccess) return false;
  atomicCAS((int*)(&errors::last_err), cudaSuccess, err);
  return true;
}

struct cudaTimer {
  cudaEvent_t start_event, stop_event;
  cudaTimer() { cudaEventCreate(&start_event); cudaEventCreate(&stop_event); }
  ~cudaTimer() { cudaEventDestroy(start_event); cudaEventDestroy(stop_event); }
  void start() { cudaEventRecord (start_event, 0);/* cudaEventSynchronize (start_event);*/ }
  float stop() {
    float res;
    cudaEventRecord (stop_event, 0); cudaEventSynchronize (stop_event);
    cudaEventElapsedTime (&res, start_event, stop_event);
    return res;
  }
  float restart() { float res=stop(); start(); return res; }
};

class cuTimer {
  cudaEvent_t tstart,tend,tlap;
  cudaStream_t st;
  float diftime,diflap;
  public:
  cuTimer(const cudaStream_t& stream=0): diftime(0),diflap(0) {
    CHECK_ERROR( cudaEventCreate(&tstart) ); 
    CHECK_ERROR( cudaEventCreate(&tend  ) );
    CHECK_ERROR( cudaEventCreate(&tlap  ) );
    CHECK_ERROR( cudaEventRecord(tstart,stream) );
    CHECK_ERROR( cudaEventRecord(tlap,stream) ); st=stream;
  }
  ~cuTimer(){
    CHECK_ERROR( cudaEventDestroy(tstart) );
    CHECK_ERROR( cudaEventDestroy(tend) );
    CHECK_ERROR( cudaEventDestroy(tlap) );
  }
  float gettime(){
    CHECK_ERROR( cudaEventRecord(tend,st) );
    CHECK_ERROR( cudaEventSynchronize(tend) );
    CHECK_ERROR( cudaEventElapsedTime(&diftime, tstart,tend) ); 
    return diftime;
  }
  float getlaptime(){
    CHECK_ERROR( cudaEventRecord(tend,st) );
    CHECK_ERROR( cudaEventSynchronize(tend) );
    CHECK_ERROR( cudaEventElapsedTime(&diflap, tlap,tend) ); 
    CHECK_ERROR( cudaEventRecord(tlap,st) );
    return diflap;
  }
};

const int CudaDevs=1;

template<class Ph, class Pd> static void copy2dev(Ph &hostP, Pd &devP) {
  if(CudaDevs>1) {
    int curdev; CHECK_ERROR( cudaGetDevice(&curdev) );
    for(int i=0; i<CudaDevs; i++) {
      CHECK_ERROR( cudaSetDevice(i) );
      CHECK_ERROR( cudaMemcpyToSymbol(devP, &hostP, sizeof(Pd)) );
    }
    CHECK_ERROR( cudaSetDevice(curdev) );
  }
  else CHECK_ERROR( cudaMemcpyToSymbol(devP, &hostP, sizeof(Pd)) );
}

//// From here https://stackoverflow.com/questions/32226300/make-variadic-macro-method-which-prints-all-variables-names-and-values
#define SHOW(...) show(std::cout, #__VA_ARGS__, __VA_ARGS__)
template<typename H1> std::ostream& show(std::ostream& out, const char* label, H1&& value) {
  return out << label << "=" << std::forward<H1>(value) << '\n';
}

template<typename H1, typename ...T> std::ostream& show(std::ostream& out, const char* label, H1&& value, T&&... rest) {
  const char* pcomma = strchr(label, ',');
  return show(out.write(label, pcomma - label) << "=" << std::forward<H1>(value) << ", ",
              pcomma + 1,
              std::forward<T>(rest)...);
}


template<class T, int N> struct CCarray: public std::array<T,N>
{
	__host__ __device__ typename std::array<T,N>::reference front() {
		return std::array<T,N>::_M_elems[0];
	}
	__host__ __device__ typename std::array<T,N>::const_reference front() const {
		return std::array<T,N>::_M_elems[0];
	}

};

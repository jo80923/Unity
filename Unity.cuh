#ifndef UNITY_CUH
#define UNITY_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  if (cudaSuccess != err) {
      fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
      file, line, cudaGetErrorString(err));
      exit(-1);
  }
#endif

  return;
}
inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
    file, line, cudaGetErrorString(err));
    exit(-1);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  // err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
    file, line, cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}

namespace jax{
  struct IllegalUnityTransition{
    std::string msg;
    IllegalUnityTransition(){
      msg = "Illegal Unity memory transfer";
    }
    IllegalUnityTransition(std::string msg) : msg("Illegal Unity memory transfer: " + msg){}
    virtual const char* what() const throw(){
      return msg.c_str();
    }
  };

  struct NullUnityException{
    std::string msg;
    NullUnityException(){
      msg = "Illegal attempt to use null set Unity";
    }
    NullUnityException(std::string msg) : msg("Illegal attempt to use null set Unity: " + msg){}
    virtual const char* what() const throw(){
      return msg.c_str();
    }
  };

  //TODO make some variable represent whether host or device versions are the most up to date
  typedef enum MemoryState{
    null = 0,
    cpu = 1,
    gpu = 2,
    both = 3,
    pinned = 4
  } MemoryState;

  template<typename T>
  class Unity{

  public:

    T* device;
    T* host;
    unsigned long numElements;
    MemoryState state;

    Unity();
    Unity(T* data, unsigned long numElements, MemoryState state);
    ~Unity();

    void setData(T* data, unsigned long numElements, MemoryState state);//hard set
    void setMemoryState(MemoryState state);//hard set
    void updateHost();//state == both
    void updateDevice();//state == both
    void clear();//hard clear
    void transferMemoryTo(MemoryState state);//soft set
    void clearHost();//hard clear
    void clearDevice();//hard clear

  };

  template<typename T>
  Unity<T>::Unity(){
    this->host = NULL;
    this->device = NULL;
    this->state = null;
    this->numElements = 0;
  }
  template<typename T>
  Unity<T>::Unity(T* data, unsigned long numElements, MemoryState state){
    this->host = NULL;
    this->device = NULL;
    this->state = state;
    this->numElements = numElements;
    if(state == cpu) this->host = data;
    else if(state == gpu) this->device = data;
    else{
      throw IllegalUnityTransition("cannot instantiate memory on device and host with only one pointer");
    }
  }
  template<typename T>
  Unity<T>::~Unity(){
    this->clear();
  }
  template<typename T>
  void Unity<T>::setData(T* data, unsigned long numElements, MemoryState state){
    this->clear();
    this->state = state;
    this->numElements = numElements;
    if(state == cpu) this->host = data;
    else if(state == gpu) this->device = data;
    else{
      throw IllegalUnityTransition("cannot instantiate memory on device and host with only one pointer");
    }
  }
  template<typename T>
  void Unity<T>::setMemoryState(MemoryState state){
    if((this->state == null || this->numElements == 0) && state != null){
      throw NullUnityException();
    }
    else if(this->state == state) return;
    else if(state == null) this->clear();
    else if(state == both){
      if(this->state == cpu){
        if(this->device == NULL){
          CudaSafeCall(cudaMalloc((void**)&this->device, sizeof(T)*this->numElements));
        }
        CudaSafeCall(cudaMemcpy(this->device,this->host, sizeof(T)*this->numElements, cudaMemcpyHostToDevice));
      }
      if(this->state == gpu){
        if(this->host == NULL){
          this->host = operator new(sizeof(T)*this->numElements);
        }
        CudaSafeCall(cudaMemcpy(this->host, this->device, sizeof(T)*this->numElements, cudaMemcpyDeviceToHost));
      }
    }
    else if(state == gpu){
      if(this->device == NULL){
        CudaSafeCall(cudaMalloc((void**)&this->device, sizeof(T)*this->numElements));
      }
      CudaSafeCall(cudaMemcpy(this->device,this->host, sizeof(T)*this->numElements, cudaMemcpyHostToDevice));
      operator delete(this->host);
      this->host = NULL;
    }
    else if(state == cpu){
      if(this->host == NULL){
        this->host = operator new(sizeof(T)*this->numElements);
      }
      CudaSafeCall(cudaMemcpy(this->host, this->device, sizeof(T)*this->numElements, cudaMemcpyDeviceToHost));
      CudaSafeCall(cudaFree(this->device));
      this->device = NULL;
    }
    else{
      throw IllegalUnityTransition("unkown memory state");
    }
    this->state = state;
  }
  template<typename T>
  void Unity<T>::updateHost(){
    if(this->state != both){
      throw IllegalUnityTransition("gpu -> cpu (update) memory state needs to be BOTH");
    }
    else{
      CudaSafeCall(cudaMemcpy(this->host, this->device, this->numElements*sizeof(T), cudaMemcpyDeviceToHost));
    }
  }
  template<typename T>
  void Unity<T>::updateDevice(){
    if(this->state != both){
      throw IllegalUnityTransition("cpu -> gpu (update) memory state needs to be BOTH");
    }
    else{
      CudaSafeCall(cudaMemcpy(this->device, this->host, this->numElements*sizeof(T), cudaMemcpyHostToDevice));
    }
  }
  template<typename T>
  void Unity<T>::clear(){
    switch(this->state){
      case null:
        break;
      case cpu:
        if(this->host != NULL){
          operator delete(this->host);
        }
        break;
      case gpu:
        if(this->device != NULL){
          CudaSafeCall(cudaFree(this->device));
        }
        break;
      case both:
        if(host != NULL){
          operator delete(this->host);
        }
        if(device != NULL){
          CudaSafeCall(cudaFree(this->device));
        }
        break;
      default:
        throw IllegalUnityTransition("unkown memory state");
    }
    this->host = NULL;
    this->device = NULL;
    this->state = null;
    this->numElements = 0;
  }
  template<typename T>
  void Unity<T>::transferMemoryTo(MemoryState state){
    if((this->state == null || sizeof(T)*this->numElements == 0) && state != null){
      throw NullUnityException();
    }
    else if(state == null){
      throw IllegalUnityTransition("Cannot transfer unity memory to null");
    }
    else if(this->state == state) return;
    else if(state == both){
      if(this->state == cpu){
        if(this->device == NULL){
          CudaSafeCall(cudaMalloc((void**)&this->device, sizeof(T)*this->numElements));
        }
        CudaSafeCall(cudaMemcpy(this->device,this->host, sizeof(T)*this->numElements, cudaMemcpyHostToDevice));
      }
      if(this->state == gpu){
        if(this->host == NULL){
          this->host = new T[this->numElements];
        }
        CudaSafeCall(cudaMemcpy(this->host, this->device, sizeof(T)*this->numElements, cudaMemcpyDeviceToHost));
      }
    }
    else if(state == gpu){
      if(this->device == NULL){
        CudaSafeCall(cudaMalloc((void**)&this->device, sizeof(T)*this->numElements));
      }
      CudaSafeCall(cudaMemcpy(this->device, this->host, sizeof(T)*this->numElements, cudaMemcpyHostToDevice));
    }
    else if(state == cpu){
      if(this->host == NULL){
        this->host = new T[this->numElements];
      }
      CudaSafeCall(cudaMemcpy(this->host, this->device, sizeof(T)*this->numElements, cudaMemcpyDeviceToHost));
    }
    else{
      throw IllegalUnityTransition("unkown memory state");
    }
    this->state = state;
  }
  template<typename T>
  void Unity<T>::clearHost(){
    if(this->host != NULL){
      operator delete(this->host);
    }
    if(this->state == cpu){
      this->state = null;
    }
    else if(this->state == both){
      this->state = gpu;
    }
  }
  template<typename T>
  void Unity<T>::clearDevice(){
    if(this->device != NULL){
      CudaSafeCall(cudaFree(this->device));
      this->device = NULL;
    }
    if(this->state == gpu){
      this->state = null;
    }
    else if(this->state == both){
      this->state = cpu;
    }
  }

}

#endif /*UNITY_CUH*/

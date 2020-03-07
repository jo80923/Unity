/**
* \file example_checkpoint_signals.cu
* \author  Jackson Parker
* \date    March 4 2020
* \brief Example on the usage of the Unity structure's checkpointing and ways to use c signals
* \details This can also act as an executable for unit testing.
* \example 
* \see example_unity.cu
* \see Unity
* \todo add signal examples for SIGABRT, SIGFPE, SIGILL, SIGSEGV, SIGTERM
*/
#include "../../include/common_includes.h"
#include "../../include/io_util.h"
#include <csignal>
#include <typeinfo>
#include <unistd.h>

/*
UTILITY
*/

template<typename T>
bool printTest(jax::Unity<T>* original, jax::Unity<T>* checkpoint){
    bool passed = true;
    jax::MemoryState origin[2] = {original->getMemoryState(), checkpoint->getMemoryState()};
    if(origin[0] != origin[1]){
        std::cout<<"state is different"<<std::endl;
        passed = false;
    }
    unsigned long size[2] = {original->size(), checkpoint->size()};
    if(size[0] != size[1]){
        std::cout<<"size is different"<<std::endl;
        return false;//should not check data then or risk segfault
    }
    if(origin[0] == jax::gpu){
        original->transferMemoryTo(jax::cpu);
    }
    if(origin[1] == jax::gpu){
        checkpoint->transferMemoryTo(jax::cpu);
    }
    int numdiff = 0;
    for(int i = 0; i < size[0]; ++i){
        if(original->host[i] != checkpoint->host[i]) numdiff++;
    }
    if(!numdiff){
        std::cout<<"data is equivalent"<<std::endl;
    }
    else{
        std::cout<<numdiff<<" elements differ"<<std::endl;
        passed = false;
    }
    return passed;
}

/*
SIGNAL HANDLING 
*/

/*
    due to the way that signals work in C++ global variables are required for 
    signal handlers to access variables
    
    NOTE: Unity does not need to be allocated with new 
    for this to work, the interrupt will just throw a device shutting 
    down error when the destructor is called if a non-pointer Unity is 
    used here - test will pass though
*/

jax::Unity<int>* i_nums;
jax::Unity<int>* i_nums_cpt;


void sigintHandler(int signal){
    std::cout<<"Interrupted by signal ("<<signal<<")"<<std::endl;
    i_nums->checkpoint(0,"./");//will write i_nums data and state information

    std::cout<<"reading checkpoint"<<std::endl;
    jax::Unity<int>* i_nums_cpt = new jax::Unity<int>("0_i.uty");

    std::cout<<"equating checkpoint data"<<std::endl;
    if(printTest(i_nums,i_nums_cpt)){
        std::cout<<"TEST PASSED"<<std::endl;
    }
    else{
        std::cout<<"TEST FAILED"<<std::endl;
    }
    delete i_nums;
    delete i_nums_cpt;
    exit(signal);
}


int main(int argc, char *argv[]){
  try{
    signal(SIGINT,sigintHandler);

    i_nums = new jax::Unity<int>(nullptr,100,jax::cpu);
    for(int i = 0; i < 100; ++i){
      i_nums->host[i] = i;
    } 
    i_nums->transferMemoryTo(jax::gpu);//check if checkpointing can handle reoriginating to both cpu and gpu with data
    
    while(true){
        std::cout<<"Waiting for interrupt to perform checkpoint test"<<std::endl;
        sleep(1);
    }

    return 0;
  }
  catch (const std::exception &e){
    std::cerr << "Caught exception: " << e.what() << '\n';
    std::exit(2);
  }
  catch (...){
    std::cerr << "Caught unknown exception\n";
    std::exit(3);
  }
}



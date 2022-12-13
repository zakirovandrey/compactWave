#include "structs.cuh"
#include "init.cu"
#include "drop.cu"
#include "update.cu"


int main(int argc, char** argv) {
  init();
  for(int it=0; it<100000; it++) {
    if(!PERFORMANCE_DEBUG && it%10==0) drop();
    printf(" >>> Run it=%d ... ",it);
    std::vector<double> timings;
    if(parsHost.Ngpus==1) calcConeFold< FarshLines<Ny/1> >(parsHost.iStep, timings); else
    if(parsHost.Ngpus==2) calcConeFold< FarshLines<Ny/2> >(parsHost.iStep, timings); else
    if(parsHost.Ngpus==3) calcConeFold< FarshLines<Ny/3> >(parsHost.iStep, timings); else
    if(parsHost.Ngpus==4) calcConeFold< FarshLines<Ny/4> >(parsHost.iStep, timings); else
    if(parsHost.Ngpus==5) calcConeFold< FarshLines<Ny/5> >(parsHost.iStep, timings); else
    if(parsHost.Ngpus==6) calcConeFold< FarshLines<Ny/6> >(parsHost.iStep, timings); else
    if(parsHost.Ngpus==7) calcConeFold< FarshLines<Ny/7> >(parsHost.iStep, timings); else
    if(parsHost.Ngpus==8) calcConeFold< FarshLines<Ny/8> >(parsHost.iStep, timings); else
	throw std::runtime_error("Unknown number of GPUs"); 
    const double msize = sizeof(Cell)*Nx*Ny*Nz*Nt*2/(1024.*1024.*1024.); //TODO
    printf("Performance: %8.2f GLU/s  | Bandwidth:%6.2f GBytes/s | \n", 1e-6*Nx*Ny*Nz*Nt/timings[0], msize/(timings[0]*1e-3) );
    parsHost.iStep++;
    for(int i=0; i<parsHost.Ngpus; i++) copy2dev( parsHost, pars );
  }

  return 0;
}

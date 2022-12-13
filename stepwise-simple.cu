#include <vector>
#include <iostream>
#include "cuda_math.h"

#define USE_FLOAT

#include "params.hpp"

#ifndef NBSIZE
#define NBSIZE 8
#endif
#ifndef NGSIZE
#define NGSIZE 1024
#endif

const int Nb=NBSIZE;
const int NSIZE=NGSIZE;//1024;//64*Nb;

const int Nx=NSIZE;
const int Ny=NSIZE;
const int Nz=NSIZE;


struct Cell{
  ftype val;
};


const ftype VelX = 1.1;
const ftype VelY = 0.2;
const ftype VelZ = 0.3;
const ftype CFL=0.01;

inline __device__ ftype get_flux_any(const ftype valM, const ftype valP, const ftype Vel, const int it) {
  
   const ftype v = Vel;
   return v*ftype(0.5)*( (valM+valP) - CFL*v*(valP-valM) ) ;;
   
}

inline __device__ ftype get_flux_X(const ftype valM, const ftype valP, const int it) {
  return get_flux_any(valM,valP,VelX,it);
}
inline __device__ ftype get_flux_Y(const ftype valM, const ftype valP, const int it) {
  return get_flux_any(valM,valP,VelY,it);
}
inline __device__ ftype get_flux_Z(const ftype valM, const ftype valP, const int it) {
  return get_flux_any(valM,valP,VelZ,it);
}


__global__ __launch_bounds__((Nb+2)*(Nb+2)*(Nb+2)) void update(Cell* c1, Cell* c2){
  const int3 thid = make_int3( threadIdx.x, threadIdx.y, threadIdx.z );

  const int3 crd0 = make_int3(blockIdx.x*Nb, blockIdx.y*Nb, blockIdx.z*Nb );

  const long3 glob_crd = make_long3( (crd0.x-1+thid.x+Nx)%Nx, (crd0.y-1+thid.y+Ny)%Ny, (crd0.z-1+thid.z+Nz)%Nz );

  const long gind = glob_crd.x + glob_crd.y*Nx + glob_crd.z*Nx*Ny;
  
  extern __shared__ Cell shc[];//(Nb+2)*(Nb+2)*(Nb+2)];

  shc[thid.x+thid.y*(Nb+2)+thid.z*(Nb+2)*(Nb+2)].val = c1[gind].val;

  __syncthreads();
  
  if(thid.x==0 || thid.y==0 || thid.z==0 || thid.x==Nb+1 || thid.y==Nb+1 || thid.z==Nb+1) return;

  auto zip_sh = [](const int x, const int y, const int z) { return shc[x+y*(Nb+2)+z*(Nb+2)*(Nb+2)].val; };

  ftype3 valM = make_ftype3( zip_sh(thid.x-1, thid.y, thid.z), zip_sh(thid.x, thid.y-1, thid.z), zip_sh(thid.x, thid.y, thid.z-1) );
  ftype3 valP = make_ftype3( zip_sh(thid.x+1, thid.y, thid.z), zip_sh(thid.x, thid.y+1, thid.z), zip_sh(thid.x, thid.y, thid.z+1) );

  const ftype valC = zip_sh(thid.x,thid.y,thid.z);

  const ftype3 fluxM = make_ftype3( get_flux_X(valM.x, valC, 0),  get_flux_Y(valM.y, valC, 0), get_flux_Z(valM.z, valC, 0) );
  const ftype3 fluxP = make_ftype3( get_flux_X(valC, valP.x, 0),  get_flux_Y(valC, valP.y, 0), get_flux_Z(valC, valP.z, 0) );

  c2[gind].val = valC + CFL* ( ( fluxM.x - fluxP.x ) + ( fluxM.y - fluxP.y ) + ( fluxM.z - fluxP.z ) );


}

Cell* c[2];

__global__ __launch_bounds__(Nb*Nb*Nb) void fill_data(Cell* c1, Cell* c2){
  const int3 thid = make_int3( threadIdx.x, threadIdx.y, threadIdx.z );

  const int3 crd0 = make_int3(blockIdx.x*Nb, blockIdx.y*Nb, blockIdx.z*Nb );
  const long3 glob_crd = make_long3( (crd0.x+thid.x+Nx)%Nx, (crd0.y+thid.y+Ny)%Ny, (crd0.z+thid.z+Nz)%Nz );

  const long gind = glob_crd.x + glob_crd.y*Nx + glob_crd.z*Nx*Ny;
  
  ftype fillval=0;
  if( length(make_ftype3(glob_crd.x-Nx/2,glob_crd.y-Ny/2,glob_crd.z-Nz/2))<Nx/4 )
     fillval = 1;
  c1[gind].val=fillval;
  c2[gind].val=fillval;
}
void init(){
	printf("Allocate memory: %.2f GB\n", 2L*sizeof(Cell)*Nx*Ny*Nz/1024./1024./1024);
  //CHECK_ERROR( cudaMallocManaged((void**)&c[0], sizeof(Cell)*Nx*Ny*Nz) ); CHECK_ERROR( cudaMemset(c[0], 0, sizeof(Cell)*Nx*Ny*Nz) );
  //CHECK_ERROR( cudaMallocManaged((void**)&c[1], sizeof(Cell)*Nx*Ny*Nz) ); CHECK_ERROR( cudaMemset(c[1], 0, sizeof(Cell)*Nx*Ny*Nz) );
  CHECK_ERROR( cudaMalloc((void**)&c[0], sizeof(Cell)*Nx*Ny*Nz) ); CHECK_ERROR( cudaMemset(c[0], 0, sizeof(Cell)*Nx*Ny*Nz) );
  CHECK_ERROR( cudaMalloc((void**)&c[1], sizeof(Cell)*Nx*Ny*Nz) ); CHECK_ERROR( cudaMemset(c[1], 0, sizeof(Cell)*Nx*Ny*Nz) );
  fill_data<<<dim3(Nx/Nb,Ny/Nb,Nz/Nb), dim3(Nb,Nb,Nb)>>>(c[0],c[1]);
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
}

void drop(const int it){
  Cell* datahost = new Cell[long(Nx)*Ny*Nz];
  CHECK_ERROR( cudaMemcpy(datahost, c[0], sizeof(Cell)*Nx*Ny*Nz, cudaMemcpyDeviceToHost) );
  char fname[256]; sprintf(fname, "Val_%07d.arr", it);
  FILE* pF; pF = fopen(fname,"w");
  int sizeofT = sizeof(ftype);
  const int dim=3;
  int comsize = sizeof(double);
  double step = 1;
  const int scale=1;//NTS;
  const int NxL = Nx/scale, NyL = Ny/scale, NzL=Nz/scale;
  fwrite(&comsize, sizeof(int  ), 1, pF);  //size of comment
  fwrite(&step   , comsize      , 1, pF);    // comment
  fwrite(&dim    , sizeof(int  ), 1, pF);     //dim =
  fwrite(&sizeofT, sizeof(int  ), 1, pF); //data size
  fwrite(&NxL     , sizeof(int  ), 1, pF);
  fwrite(&NyL     , sizeof(int  ), 1, pF);
  fwrite(&NzL     , sizeof(int  ), 1, pF);
  //printf("saving %s\n",fname);
  for(int z=0; z<NzL; z++) for(int y=0; y<NyL; y++) for(int x=0; x<NxL; x++) {
    const int3 crd = make_int3(x,y,z)*scale;
    ftype val = datahost[crd.x + crd.y*Nx + crd.z*Nx*Ny].val;
    fwrite(&val, sizeof(ftype), 1, pF);
    if(x==0 && y==0) { printf(" Drop data ... progress = %2d%%\r", (z+1)*100/NzL); fflush(stdout); }
  }
  std::cout<< "Drop data OK" << std::endl;
  fclose(pF);
  delete[] datahost;
}

static const size_t sh_size = (Nb+2)*(Nb+2)*(Nb+2)*sizeof(Cell);

void calcStep(const int it, std::vector<double>& timings) {
  cuTimer t0;
  
  static_assert(Nx%Nb==0 && Ny%Nb==0 && Nz%Nb==0);

  update<<<dim3(Nx/Nb,Ny/Nb,Nz/Nb), dim3(Nb+2,Nb+2,Nb+2), sh_size>>>(c[0],c[1]);
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );

  std::swap(c[0],c[1]);

  timings.push_back( t0.getlaptime() );
}

int main(int argc, char** argv) {
  init();
  CHECK_ERROR(cudaFuncSetAttribute(update, cudaFuncAttributeMaxDynamicSharedMemorySize, sh_size));
  for(int it=0; it<10; it++) {
    //if(it%100==0) drop(it);
    printf(" >>> Run it=%d ... ",it);
    std::vector<double> timings;
    calcStep(it, timings);
    const double msize = 0*double(1)*Nx*Ny*Nz*sizeof(Cell)/1024./1024./1024.;
    printf("Performance: %8.2f GLU/s  | Bandwidth:%6.2f GBytes/s | \n", 1e-6*Nx*Ny*Nz/timings[0], msize/(timings[0]*1e-3) );
  }

  return 0;
}

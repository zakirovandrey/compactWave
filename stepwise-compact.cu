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
const int NSIZE=NGSIZE;//64*Nb;

const int Nx=NSIZE;
const int Ny=NSIZE;
const int Nz=NSIZE;

const int Nt=64;

struct Cell{
  ftype val;
  ftype3 fluxes;
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


template<int> __global__ __launch_bounds__(Nb*Nb*Nb) void updateFluxes(Cell* c){
  const int3 thid = make_int3( threadIdx.x, threadIdx.y, threadIdx.z );

  const int3 crd0 = make_int3(blockIdx.x*Nb, blockIdx.y*Nb, blockIdx.z*Nb );

  //const int3 lcind = make_int3( thid.x*2+(i&1) ,  (thid.y*2+(i>>1&1) ,  (thid.z*2+(i>>2&1)) );
  const int3 lcind = make_int3( thid.x, thid.y, thid.z );
  const long3 glob_crd = make_long3( (crd0.x+lcind.x+Nx)%Nx, (crd0.y+lcind.y+Ny)%Ny, (crd0.z+lcind.z+Nz)%Nz );

  const long gind = glob_crd.x + glob_crd.y*Nx + glob_crd.z*Nx*Ny;
  
  __shared__ ftype shc[Nb*Nb*Nb];

  shc[ lcind.x + lcind.y*Nb + lcind.z*Nb*Nb ] = c[gind].val;

  __syncthreads();

  auto zip_sh = [](const int x,const int y, const int z) { return shc[x+y*Nb+z*Nb*Nb]; };
  const ftype valC = zip_sh(lcind.x,lcind.y,lcind.z);

  if(thid.x%2==0) c[gind].fluxes.x = get_flux_X(valC, zip_sh(lcind.x+1,lcind.y,lcind.z), 0);
  else            c[gind].fluxes.x = get_flux_X(zip_sh(lcind.x-1,lcind.y,lcind.z), valC, 0);
  if(thid.y%2==0) c[gind].fluxes.y = get_flux_Y(valC, zip_sh(lcind.x,lcind.y+1,lcind.z), 0);
  else            c[gind].fluxes.y = get_flux_Y(zip_sh(lcind.x,lcind.y-1,lcind.z), valC, 0);
  if(thid.z%2==0) c[gind].fluxes.z = get_flux_Z(valC, zip_sh(lcind.x, lcind.y, lcind.z+1), 0);
  else            c[gind].fluxes.z = get_flux_Z(zip_sh(lcind.x,lcind.y,lcind.z-1), valC, 0);
}

template<int parity, const int CALC_FLUXES_AGAIN=0> __global__ __launch_bounds__(Nb*Nb*Nb) void updateVals(Cell* c){
  const int3 thid = make_int3( threadIdx.x, threadIdx.y, threadIdx.z );

  const int3 crd0 = make_int3((blockIdx.x*Nb+parity)%Nx, (blockIdx.y*Nb+parity)%Ny, (blockIdx.z*Nb+parity)%Nz );

  //const int3 lcind = make_int3( thid.x*2+(i&1) ,  (thid.y*2+(i>>1&1) ,  (thid.z*2+(i>>2&1)) );
  const int3 lcind = make_int3( thid.x, thid.y, thid.z );
  const long3 glob_crd = make_long3( (crd0.x+lcind.x+Nx)%Nx, (crd0.y+lcind.y+Ny)%Ny, (crd0.z+lcind.z+Nz)%Nz );

  const long gind = glob_crd.x + glob_crd.y*Nx + glob_crd.z*Nx*Ny;
  
  __shared__ Cell shc[Nb*Nb*Nb];

  shc[ lcind.x + lcind.y*Nb + lcind.z*Nb*Nb ] = c[gind];

  __syncthreads();

  auto zip_sh = [](const int x,const int y, const int z) { return shc[x+y*Nb+z*Nb*Nb].val; };
  const Cell loc_cell = shc[lcind.x+lcind.y*Nb+lcind.z*Nb*Nb];

  ftype3 t_fluxes;

  if(thid.x%2==0) t_fluxes.x = get_flux_X(loc_cell.val, zip_sh(lcind.x+1,lcind.y,lcind.z), 0);
  else            t_fluxes.x = get_flux_X(zip_sh(lcind.x-1,lcind.y,lcind.z), loc_cell.val, 0);
  if(thid.y%2==0) t_fluxes.y = get_flux_Y(loc_cell.val, zip_sh(lcind.x, lcind.y+1,lcind.z), 0);
  else            t_fluxes.y = get_flux_Y(zip_sh(lcind.x, lcind.y-1,lcind.z), loc_cell.val, 0);
  if(thid.z%2==0) t_fluxes.z = get_flux_Z(loc_cell.val, zip_sh(lcind.x,lcind.y,lcind.z+1), 0);
  else            t_fluxes.z = get_flux_Z(zip_sh(lcind.x,lcind.y,lcind.z-1), loc_cell.val, 0);

  const ftype new_val = loc_cell.val + CFL*(
		  (loc_cell.fluxes.x - t_fluxes.x)*(1-(thid.x&1)*2) + 
		  (loc_cell.fluxes.y - t_fluxes.y)*(1-(thid.y&1)*2) + 
		  (loc_cell.fluxes.z - t_fluxes.z)*(1-(thid.z&1)*2)
		);
  c[gind].val = new_val;

  if(!CALC_FLUXES_AGAIN) return;

  __syncthreads();

  shc[lcind.x + lcind.y*Nb + lcind.z*Nb*Nb].val = new_val;
  
  __syncthreads();

  const ftype valC = zip_sh(lcind.x,lcind.y,lcind.z);

  if(thid.x%2==0) c[gind].fluxes.x = get_flux_X(valC, zip_sh(lcind.x+1,lcind.y,lcind.z), 0);
  else            c[gind].fluxes.x = get_flux_X(zip_sh(lcind.x-1,lcind.y,lcind.z), valC, 0);
  if(thid.y%2==0) c[gind].fluxes.y = get_flux_Y(valC, zip_sh(lcind.x,lcind.y+1,lcind.z), 0);
  else            c[gind].fluxes.y = get_flux_Y(zip_sh(lcind.x,lcind.y-1,lcind.z), valC, 0);
  if(thid.z%2==0) c[gind].fluxes.z = get_flux_Z(valC, zip_sh(lcind.x, lcind.y, lcind.z+1), 0);
  else            c[gind].fluxes.z = get_flux_Z(zip_sh(lcind.x,lcind.y,lcind.z-1), valC, 0);


}

__global__ __launch_bounds__(Nb*Nb*Nb) void fill_data(Cell* c){
  const int3 thid = make_int3( threadIdx.x, threadIdx.y, threadIdx.z );

  const int3 crd0 = make_int3(blockIdx.x*Nb, blockIdx.y*Nb, blockIdx.z*Nb );
  const long3 glob_crd = make_long3( (crd0.x+thid.x+Nx)%Nx, (crd0.y+thid.y+Ny)%Ny, (crd0.z+thid.z+Nz)%Nz );

  const long gind = glob_crd.x + glob_crd.y*Nx + glob_crd.z*Nx*Ny;
  
  if( length(make_ftype3(glob_crd.x-Nx/2,glob_crd.y-Ny/2,glob_crd.z-Nz/2))<Nx/4 )
	  c[gind].val=1;
  else c[gind].val=0;
}
Cell* cglob;

void init(){
  printf("Allocate memory: %.2f GB\n", sizeof(Cell)*Nx*Ny*Nz/1024./1024./1024);
  //CHECK_ERROR( cudaMallocManaged((void**)&cglob, sizeof(Cell)*Nx*Ny*Nz) ); CHECK_ERROR( cudaMemset(cglob, 0, sizeof(Cell)*Nx*Ny*Nz) );
  CHECK_ERROR( cudaMalloc((void**)&cglob, sizeof(Cell)*Nx*Ny*Nz) ); CHECK_ERROR( cudaMemset(cglob, 0, sizeof(Cell)*Nx*Ny*Nz) );
  fill_data<<<dim3(Nx/Nb,Ny/Nb,Nz/Nb), dim3(Nb,Nb,Nb)>>>(cglob);
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
}
void drop(const int it){
  Cell* datahost = new Cell[long(1)*Nx*Ny*Nz];
  CHECK_ERROR( cudaMemcpy(datahost, cglob, sizeof(Cell)*Nx*Ny*Nz, cudaMemcpyDeviceToHost) );
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


void calcStep(const int it, std::vector<double>& timings, int& Niters) {
  cuTimer t0;
  
  static_assert(Nx%Nb==0 && Ny%Nb==0 && Nz%Nb==0);
  assert(Nb%2==0);
  Niters=0;

  /*
  updateFluxes<0><<<dim3(Nx/Nb,Ny/Nb,Nz/Nb), dim3(Nb,Nb,Nb)>>>(cglob); cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  updateVals<1><<<dim3(Nx/Nb,Ny/Nb,Nz/Nb), dim3(Nb,Nb,Nb)>>>(cglob);   cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  Niters++;
  */
  
  updateFluxes<0><<<dim3(Nx/Nb,Ny/Nb,Nz/Nb), dim3(Nb,Nb,Nb)>>>(cglob); cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  for(int it=0;it<Nt-1;it++) {
    if(it%2==0) updateVals<0,1><<<dim3(Nx/Nb,Ny/Nb,Nz/Nb), dim3(Nb,Nb,Nb)>>>(cglob);
    else        updateVals<1,1><<<dim3(Nx/Nb,Ny/Nb,Nz/Nb), dim3(Nb,Nb,Nb)>>>(cglob);
    cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
    Niters++;
  }
  updateVals<Nt%2><<<dim3(Nx/Nb,Ny/Nb,Nz/Nb), dim3(Nb,Nb,Nb)>>>(cglob); cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  Niters++;

  timings.push_back( t0.getlaptime() );
}

int main(int argc, char** argv) {
  init();
  for(int it=0; it<10; it++) {
//    if(it%100==0) drop(it);
    printf(" >>> Run it=%d ... ",it);
    std::vector<double> timings;
    int Niters=0;
    calcStep(it, timings, Niters);
    const double msize = 0*double(1)*Nx*Ny*Nz*sizeof(Cell)/1024./1024./1024.;
    printf("Performance: %8.2f GLU/s  | Bandwidth:%6.2f GBytes/s | \n", 1e-6*Nx*Ny*Nz*Niters/timings[0], msize/(timings[0]*1e-3) );
  }

  return 0;
}

#include <iostream>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "structs.cuh"

__global__ void fill(const int ix, Tile* buf);

void alloc_mmaped(void** ptr, const size_t size);

int init(){
  parsHost.iStep=0;
  printf("Data allocation %d x %d x %d (%.2f GB)\n", Nx, Ny, Nz, sizeof(Data)/1024./1024./1024.);
  CHECK_ERROR( cudaMallocHost((void**)&parsHost.data, sizeof(Data)) );
  //CHECK_ERROR( cudaMallocManaged((void**)&parsHost.data, sizeof(Data)) );
  //CHECK_ERROR( cudaMalloc((void**)&parsHost.data, sizeof(Data)) );
  //alloc_mmaped((void**)&parsHost.data, sizeof(Data));
  std::cout<< "Data memset..." << std::endl;
  //CHECK_ERROR( cudaMemset(parsHost.data, 0, sizeof(Data)) );
  CHECK_ERROR( cudaGetDeviceCount(&parsHost.Ngpus) );
  printf("Using %d GPUs\n", parsHost.Ngpus);
  assert( Ny%parsHost.Ngpus == 0);
  assert( (Ny/parsHost.Ngpus)%2 == 0);
  if(parsHost.Ngpus==1) parsHost.farshsize = sizeof(FarshLines<Ny>); else
  if(parsHost.Ngpus==2) parsHost.farshsize = sizeof(FarshLines<Ny/2>); else
  if(parsHost.Ngpus==3) parsHost.farshsize = sizeof(FarshLines<Ny/3>); else
  if(parsHost.Ngpus==4) parsHost.farshsize = sizeof(FarshLines<Ny/4>); else
  if(parsHost.Ngpus==5) parsHost.farshsize = sizeof(FarshLines<Ny/5>); else
  if(parsHost.Ngpus==6) parsHost.farshsize = sizeof(FarshLines<Ny/6>); else
  if(parsHost.Ngpus==7) parsHost.farshsize = sizeof(FarshLines<Ny/7>); else
  if(parsHost.Ngpus==8) parsHost.farshsize = sizeof(FarshLines<Ny/8>); else
  throw std::runtime_error( "Incorrect GPUs number" );
  for(int igpu = 0; igpu<parsHost.Ngpus; igpu++) {
	CHECK_ERROR(cudaSetDevice(igpu));
    printf("Farsh Memory allocation : %.2f GB on every GPU\n", parsHost.farshsize/1024./1024./1024.);
    CHECK_ERROR( cudaMalloc((void**)&(parsHost.farsh[igpu]), parsHost.farshsize) );
    std::cout<< "Farsh memset..." << std::endl;
    CHECK_ERROR( cudaMemset(parsHost.farsh[igpu], 0, parsHost.farshsize) );
    copy2dev( parsHost, pars );
  }
  for(int igpu = 0; igpu<parsHost.Ngpus; igpu++) {
	CHECK_ERROR(cudaSetDevice(igpu));
    copy2dev( parsHost, pars );
  }
  std::cout<< "Fill data..." << std::endl;
  assert(Nx%32==0 && Ny%4==0 && Nz%32==0);
  std::cout << "Progress ..."; 
  Tile* buffer; CHECK_ERROR( cudaMallocManaged((void**)&buffer, sizeof(Tile)) );
  for(int ix=0;ix<Nx;ix++) {
	  printf(" %3d%\b\b\b\b\b", int(ix*100./Nx)); fflush(stdout);
	  cudaPointerAttributes ptrprop;
	  CHECK_ERROR( cudaPointerGetAttributes( &ptrprop, parsHost.data ));
	  if( ptrprop.type!=cudaMemoryTypeDevice && ptrprop.type!=cudaMemoryTypeManaged ) {
		  fill<<<dim3(Ny/4,Nz/32), dim3(4,32)>>>(ix, buffer); cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
		  CHECK_ERROR( cudaMemcpy(&parsHost.data->tiles[ix], buffer, sizeof(Tile), cudaMemcpyDefault));
		  madvise(&parsHost.data->tiles[ix], sizeof(Tile), MADV_DONTNEED);
	  } else {
		  fill<<<dim3(Ny/4,Nz/32), dim3(4,32)>>>(ix, &parsHost.data->tiles[ix]); cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
	  }
  }
  CHECK_ERROR( cudaFree(buffer) );
  std::cout << "Init Ok" << std::endl;
  return 0;
}

template<typename T> inline __host__ __device__ T sech(const T val) { return 1/cosh(val); }

__global__ void fill(const int ix, Tile* buffer){
	if(PERFORMANCE_DEBUG) return;
	const int3 crd = make_int3(ix, blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y);
	assert(crd.x<Nx && crd.y<Ny && crd.z<Nz);
	
	//ftype& v = pars.data->tiles[crd.x].val[crd.y][crd.z];
	carray<Ncomp>& v = buffer->val[crd.y][crd.z];
	const ftype rl = length( make_ftype3(crd.x-Nx/2, crd.y-Ny/2, crd.z-Nz/2)) ;
	//for(int icomp=0;icomp<Ncomp;icomp++) v.elem[icomp]=exp(-rl*rl/5); return;
	//v.elem[0] = exp(-rl*rl/5);
	//if(rl<10) v.elem[0] = 2*M_PI;

	const ftype m=1;
	const ftype vel=0.1;
	const ftype delta=0;
	const ftype gamma = -sqrt(1/(1-vel*vel));

	const ftype3 r = make_ftype3(crd.x-Nx/2, crd.y-Ny/2, crd.z-Nz/2) ;
	const ftype mu=0.1;
	for(int icomp=0;icomp<Ncomp;icomp++) {
	    const ftype koef = mu/sqrt(1-mu*mu)*sin(sqrt(1-mu*mu)*(-icomp));
		v.elem[icomp] = 4*4*4*atan(koef*sech(mu*r.x))*atan(koef*sech(mu*r.y))*atan(koef*sech(mu*r.z));
	}
	for(int icomp=0;icomp<Ncomp;icomp++) {
		const ftype w=0.9;//0.999;
		const ftype koef = sqrt(1-w*w)/w*sin(w*(-icomp)*dt);
		//v.elem[icomp] = 4*4*4*atan(koef*sech(r.x*sqrt(1-w*w)))*atan(koef*sech(r.y*sqrt(1-w*w)))*atan(koef*sech(r.z*sqrt(1-w*w)));
		v.elem[icomp] = 4*atan(koef*sech(r.x*sqrt(1-w*w))*sech(r.y*sqrt(1-w*w))*sech(r.z*sqrt(1-w*w)));
	}
	for(int icomp=0;icomp<Ncomp;icomp++) {
      //if(crd.x==Nx/2 && crd.y==Ny/2 && crd.z==Nz/2) v.elem[icomp] =1; else v.elem[icomp] =0;
	}
}

void alloc_mmaped(void** ptr, const size_t size) {
	char swapfname[1024]; sprintf(swapfname, "tmp.swp");
	int swp_file; swp_file = open(swapfname,O_RDWR|O_TRUNC|O_CREAT, 0666);
	if(swp_file==-1) throw std::runtime_error( "Error opening file "+std::string(swapfname) );
	lseek(swp_file, size, SEEK_SET);
	write(swp_file, "", 1); lseek(swp_file, 0, SEEK_SET);
	*ptr = (void*)mmap(0, size, PROT_READ|PROT_WRITE, MAP_SHARED, swp_file,0);
	if(*ptr == MAP_FAILED) throw std::runtime_error("Error mmap data");
	close(swp_file);
}	

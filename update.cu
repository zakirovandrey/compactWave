#include <vector>
#include <thrust/swap.h>

#define PRINT_VAL(val) << ", " #val "= " << val

const int DEBUG=0;

#define SET_BC_VAL
//#define SET_BC_FLUX

template<typename farshType, const int parity>  __global__ __launch_bounds__(Nz) void compactTower(int ix_base, farshType*);
template<typename farshType>  __global__ __launch_bounds__(Nz) void copy_mem2farsh(const int fix, farshType* farsh_ptr, const Tile* buffer, const int iyshift);
template<typename farshType>  __global__ __launch_bounds__(Nz) void copy_farsh2mem(const int fix, const farshType* farsh_ptr, Tile* buffer, const int iyshift);
	
template<typename Ftype> inline __device__ Ftype get_flux_X(const Ftype valM, const Ftype valP, const int it);
template<typename Ftype> inline __device__ Ftype get_flux_Y(const Ftype valM, const Ftype valP, const int it);
template<typename Ftype> inline __device__ Ftype get_flux_Z(const Ftype valM, const Ftype valP, const int it);

const ftype CFL=dt*dt/dx*dx;
const ftype VelX=1.0;//0.9;//0.9;//0.9;
const ftype VelY=0.1;//-0.08;//-0.8;
const ftype VelZ=0.2;//-0.07;//-0.7;

inline __host__ __device__ bool isOutX(const int x) { return (x<0 || x>=Nx); }

template<typename farshType> void calcConeFold(int it, std::vector<double>& timings){
  cuTimer t0;
  for(int igpu=0; igpu<parsHost.Ngpus; igpu++) {
	  CHECK_ERROR( cudaSetDevice(igpu) );
	  CHECK_ERROR( cudaMemset(parsHost.farsh[igpu], 0, sizeof(farshType)) );
  }
  size_t sh_size = Nz*2*sizeof(ftype);
  CHECK_ERROR(cudaFuncSetAttribute(compactTower<farshType,0>, cudaFuncAttributeMaxDynamicSharedMemorySize, sh_size));
  CHECK_ERROR(cudaFuncSetAttribute(compactTower<farshType,1>, cudaFuncAttributeMaxDynamicSharedMemorySize, sh_size));
  cudaStream_t streams[parsHost.Ngpus];
  Tile* buffer[8]={0};
  const size_t bufsize = sizeof(Tile)/parsHost.Ngpus;
  for(int i=0; i<parsHost.Ngpus; i++) {
	  CHECK_ERROR( cudaSetDevice(i) );
	  CHECK_ERROR( cudaMalloc( (void**)&(buffer[i]), bufsize ) );
	  CHECK_ERROR( cudaStreamCreate(&streams[i]) );
  }
  const int NFL = Farsh::NFL;
  CellLine<NFL>* tmpline=0;
  if(parsHost.Ngpus>1) CHECK_ERROR( cudaMalloc((void**)&tmpline, sizeof(CellLine<NFL>)) );
  for(int ix=Nx-1; ix>=-Nt; ix--) {
	//printf("ix=%d\n",ix);
	for(int igpu=0; igpu<parsHost.Ngpus; igpu++) {  
		CHECK_ERROR( cudaSetDevice(igpu) );
		const int iyg = (farshType::NFY*igpu+ix+Nt)%Ny;
		const size_t bufpart1 = bufsize/farshType::NFY*(Ny-iyg);
		const size_t bufpart2 = bufsize-bufpart1;
		if(DEBUG) { std::cout << "DEBUG: ";  SHOW(igpu,ix,iyg); fflush(stdout); }
		if( !isOutX(ix) ) {
			if( iyg>Ny-farshType::NFY ) {
				CHECK_ERROR( cudaMemcpyAsync( buffer[igpu]                  , &parsHost.data->tiles[ix].val[iyg][0], bufpart1, cudaMemcpyDefault, streams[igpu]) );
				CHECK_ERROR( cudaMemcpyAsync( (char*)(buffer[igpu])+bufpart1, &parsHost.data->tiles[ix].val[0  ][0], bufpart2, cudaMemcpyDefault, streams[igpu]) );
			} else
				CHECK_ERROR( cudaMemcpyAsync( buffer[igpu]                  , &parsHost.data->tiles[ix].val[iyg][0], bufsize , cudaMemcpyDefault, streams[igpu]) );
		}
	}
	for(int igpu=0; igpu<parsHost.Ngpus; igpu++) {  
		CHECK_ERROR( cudaSetDevice(igpu) );
        farshType* farsh_ptr = static_cast<farshType*>(parsHost.farsh[igpu]);
		const int iyg = (farshType::NFY*igpu+ix+Nt)%Ny;
		const int iyf = iyg%farshType::NFY;
		if( !isOutX(ix) ) copy_mem2farsh<farshType><<<farshType::NFY,Nz, 0, streams[igpu]>>>( ix%NFL       , farsh_ptr, buffer[igpu], iyf);
		
		if(( ix+Nt)%2==0 ) compactTower<farshType,0> <<< farshType::NFY/2, Nz, sh_size, streams[igpu] >>> (ix, farsh_ptr);
		else               compactTower<farshType,1> <<< farshType::NFY/2, Nz, sh_size, streams[igpu] >>> (ix, farsh_ptr);
		
		if( !isOutX(ix+Nt+1) ) copy_farsh2mem<farshType><<<farshType::NFY,Nz, 0, streams[igpu]>>>((ix+Nt+1)%NFL, farsh_ptr, buffer[igpu], iyf);
	}
	for(int igpu=0; igpu<parsHost.Ngpus; igpu++) {  
		CHECK_ERROR( cudaSetDevice(igpu) );
		const int iyg = (farshType::NFY*igpu+ix+Nt)%Ny;
		const size_t bufpart1 = bufsize/farshType::NFY*(Ny-iyg);
		const size_t bufpart2 = bufsize-bufpart1;
		if( !isOutX(ix+Nt+1) ) {
			if( iyg>Ny-farshType::NFY ) {
				CHECK_ERROR( cudaMemcpyAsync( &parsHost.data->tiles[ix+Nt+1].val[iyg][0], buffer[igpu]                  , bufpart1, cudaMemcpyDefault, streams[igpu]) );
				CHECK_ERROR( cudaMemcpyAsync( &parsHost.data->tiles[ix+Nt+1].val[0  ][0], (char*)(buffer[igpu])+bufpart1, bufpart2, cudaMemcpyDefault, streams[igpu]) );
			} else
				CHECK_ERROR( cudaMemcpyAsync( &parsHost.data->tiles[ix+Nt+1].val[iyg][0], buffer[igpu]                  , bufsize, cudaMemcpyDefault, streams[igpu]) );
		}
	}
	// cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
	// mix-rotate farsh lines
	if(parsHost.Ngpus>1) {
		CHECK_ERROR( cudaSetDevice(parsHost.Ngpus-1) );
		for(int igpu=parsHost.Ngpus; igpu>=0; igpu--) {
			const int iygnext = (farshType::NFY*igpu+ix+Nt-1+Ny)%Ny;
			const int iyfnext = iygnext%farshType::NFY;
			CellLine<NFL>* from_ptr = tmpline, *to_ptr=tmpline;
			if(igpu>0             ) from_ptr = &static_cast<farshType*>(parsHost.farsh[igpu-1])->cls[iyfnext];
			if(igpu<parsHost.Ngpus) to_ptr   = &static_cast<farshType*>(parsHost.farsh[igpu  ])->cls[iyfnext];
			CHECK_ERROR( cudaMemcpyAsync( to_ptr, from_ptr, sizeof(CellLine<NFL>), cudaMemcpyDefault, streams[max(igpu-1,0)]) );
			CHECK_ERROR( cudaStreamSynchronize( streams[max(igpu-1,0)] ) );
		}
	}

	cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
	if( ix+Nt+1>=0 && ix+Nt+1<Nx ) madvise(&parsHost.data->tiles[ix+Nt+1], sizeof(Tile), MADV_DONTNEED);
  }
  for(int i=0;i<parsHost.Ngpus;i++) {
	CHECK_ERROR( cudaSetDevice(i) );
	CHECK_ERROR( cudaFree(buffer[i]) );
	CHECK_ERROR( cudaStreamDestroy(streams[i]) );
  }
  if(parsHost.Ngpus>1) CHECK_ERROR( cudaFree(tmpline) );
  timings.push_back( t0.getlaptime() );
}

inline __device__ void compact_step(Cell c[4], ftype* vals_sh, const int sub_it);
 
template<typename farshType>  __global__ __launch_bounds__(Nz) void copy_mem2farsh(const int fix, farshType* farsh_ptr, const Tile* buffer, const int iyshift){
	const int iy=blockIdx.x;
	const int iz=threadIdx.x;
	farsh_ptr->cls[(iy+iyshift)%farshType::NFY].valflux[ fix ][iz] = buffer->val[iy][iz].elem2;
}
template<typename farshType>  __global__ __launch_bounds__(Nz) void copy_farsh2mem(const int fix, const farshType* farsh_ptr, Tile* buffer, const int iyshift){
	const int iy=blockIdx.x;
	const int iz=threadIdx.x;
	int igpu; cudaGetDevice(&igpu);
	buffer->val[iy][iz].elem2 = farsh_ptr->cls[(iy+iyshift)%farshType::NFY].valflux[ fix ][iz];
}
template<typename farshType, const int parity>  __global__ __launch_bounds__(Nz) void compactTower(int ix, farshType* farshptr){
  const int iz = threadIdx.x;
  const int iy = (blockIdx.x*2+parity)%farshType::NFY;
  const int iyP = (iy+1)%farshType::NFY;
  extern __shared__ ftype vals_sh[2*Nz];
  register union {
	  Cell c[4] = {0};
	  ftype2 vrhs[4];
  };
  const int NFL = Farsh::NFL;
  // load_four //
  if(!isOutX(ix  )) vrhs[0] = farshptr->cls[iy ].valflux[ ix%NFL   ][iz];
  if(!isOutX(ix+1)) vrhs[1] = farshptr->cls[iy ].valflux[(ix+1)%NFL][iz];
  if(!isOutX(ix  )) vrhs[2] = farshptr->cls[iyP].valflux[ ix%NFL   ][iz];
  if(!isOutX(ix+1)) vrhs[3] = farshptr->cls[iyP].valflux[(ix+1)%NFL][iz];

  for(int it=max(0,-ix-2); it<min(Nt,Nx-ix); it++) {
	vals_sh[iz   ] = c[1].val;
	vals_sh[iz+Nz] = c[3].val;
     __syncthreads();
    compact_step(c, vals_sh, it);
     __syncthreads();
	// save-1
	if(!isOutX(ix+it)) {
		farshptr->cls[iy ].valflux[(ix+it)%NFL][iz] = vrhs[0];
	    farshptr->cls[iyP].valflux[(ix+it)%NFL][iz] = vrhs[2];
	}
	//	shift //
	c[0] = c[1];
	c[2] = c[3];
	// load-1
    if(!isOutX(ix+2+it)) {
		vrhs[1] = farshptr->cls[iy ].valflux[(ix+2+it)%NFL][iz];
	    vrhs[3] = farshptr->cls[iyP].valflux[(ix+2+it)%NFL][iz];
	}
  }
  if(!isOutX(ix+Nt)) {
    farshptr->cls[iy ].valflux[(ix+Nt)%NFL][iz] = vrhs[0];
    farshptr->cls[iyP].valflux[(ix+Nt)%NFL][iz] = vrhs[2];
  }
}

inline __device__ void compact_step(Cell c[4], ftype* vals_sh, const int sub_it){
  const int it=pars.iStep*Nt+sub_it;
  const int iz=threadIdx.x;
  //const ftype flux02 = get_flux_Y( c[0].val, c[2].val, it );
  //const ftype flux01 = get_flux_X( c[0].val, c[1].val, it );
  //const ftype flux23 = get_flux_X( c[2].val, c[3].val, it );
  //const ftype flux13 = get_flux_Y( c[1].val, c[3].val, it );
  //c[0].rhs+= (+flux02+flux01)*CFL;
  //c[2].rhs+= (-flux02+flux23)*CFL;
  c[0].rhs+= (-c[2].val-c[1].val)*CFL;
  c[2].rhs+= (-c[0].val-c[3].val)*CFL;
  //const ftype difluxZ1 = get_flux_Z( c[1].val, vals_sh[(iz+1)%Nz   ], it ) - get_flux_Z( vals_sh[(iz-1+Nz)%Nz   ], c[1].val, it );
  //const ftype difluxZ3 = get_flux_Z( c[3].val, vals_sh[(iz+1)%Nz+Nz], it ) - get_flux_Z( vals_sh[(iz-1+Nz)%Nz+Nz], c[3].val, it );
  const ftype c1Zm = vals_sh[(iz-1+Nz)%Nz   ], c1Zp = vals_sh[(iz+1)%Nz   ];
  const ftype c3Zm = vals_sh[(iz-1+Nz)%Nz+Nz], c3Zp = vals_sh[(iz+1)%Nz+Nz];
  ftype prev_val1=c[1].val, prev_val3=c[3].val;
  c[1].rhs+= (-c[0].val - c[3].val - c1Zp - c1Zm + 6*c[1].val)*CFL + dt2*sin(c[1].val);
  c[3].rhs+= (-c[2].val - c[1].val - c3Zp - c3Zm + 6*c[3].val)*CFL + dt2*sin(c[3].val);
  c[1].val = 2*c[1].val - c[1].rhs; c[1].rhs = prev_val1;
  c[3].val = 2*c[3].val - c[3].rhs; c[3].rhs = prev_val3;

	 // (-flux01+flux13+difluxZ1)*CFL + dt2*sin(c[1].val); prev_val = c[1].val; c[1].val = 2*c[1].val - c[1].rhs; c[1].rhs = prev_val;
  //c[3].rhs+= 
     //(-flux23-flux13+difluxZ3)*CFL + dt2*sin(c[3].val); prev_val = c[3].val; c[3].val = 2*c[3].val - c[3].rhs; c[3].rhs = prev_val;
}


template<typename Ftype> inline __device__ Ftype get_flux_any(const Ftype valM, const Ftype valP, const ftype Vel, const int it) {
    return -(valP-valM);
	// : Ugolok
//   return Vel*((Vel>0)?valM:valP);

//  const ftype v = ((it%2)?1.0:-1.0);
//  return v*(v>0?valM:valP);

//   const ftype v = Vel;
//   return v*ftype(0.5)*( (valM+valP) - CFL*v*(valP-valM) ) ;;

 // : Lax-Vendorff convection-diffusion  
	const Ftype v = (valP+valM)*ftype(0.5)*Vel;
	const Ftype Dcoff = 0.1;
	return v*ftype(0.5)*( (valM+valP) - CFL*v*(valP-valM) ) - Dcoff*(valP-valM);
  
  // ::: Heat diffusion
  // const ftype Dcoff = 0.1;
  // return -Dcoff*(valP-valM);
}

template<typename Ftype> inline __device__ Ftype get_flux_X(const Ftype valM, const Ftype valP, const int it) {
  return get_flux_any(valM,valP,VelX,it);
}
template<typename Ftype> inline __device__ Ftype get_flux_Y(const Ftype valM, const Ftype valP, const int it) {
  return get_flux_any(valM,valP,VelY,it);
}
template<typename Ftype> inline __device__ Ftype get_flux_Z(const Ftype valM, const Ftype valP, const int it) {
  return get_flux_any(valM,valP,VelZ,it);
}

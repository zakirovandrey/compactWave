#include <stdio.h> 

int dropScale();
int dropChunk();

int drop(){
	return dropScale();
	//return dropChunk():
}

int dropScale(){
//	return 0;
  Data* datahost = 0;
  cudaPointerAttributes ptrprop;
  CHECK_ERROR( cudaPointerGetAttributes( &ptrprop, parsHost.data ));
  if( ptrprop.type==cudaMemoryTypeHost         ) datahost = parsHost.data;
  if( ptrprop.type==cudaMemoryTypeDevice       ) { datahost=new Data; CHECK_ERROR( cudaMemcpy(datahost, parsHost.data, sizeof(Data), cudaMemcpyDefault) ); }
  if( ptrprop.type==cudaMemoryTypeManaged      ) { datahost=new Data; CHECK_ERROR( cudaMemcpy(datahost, parsHost.data, sizeof(Data), cudaMemcpyDefault) ); }
  if( ptrprop.type==cudaMemoryTypeUnregistered ) { datahost=new Data; memcpy(datahost, parsHost.data, sizeof(Data)); }
  char fname[256]; sprintf(fname, "Val_%07d.arr", parsHost.iStep);
  FILE* pF; pF = fopen(fname,"w");
  int sizeofT = sizeof(ftype);
  const int dim=3;
  int comsize = sizeof(double);
  double step = 1;
  const int scale=1;
  const int NzL = Nz/scale, NyL = Ny/scale, NxL=Nx/scale;
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
    carray<Ncomp> val = datahost->tiles[crd.x].val[crd.y][crd.z];
    fwrite(&val.elem[0], sizeof(ftype), 1, pF);
    if(x==0 && y==0) { printf(" Drop data ... progress = %2d%%       \r", (z+1)*100/NzL); fflush(stdout); }
  }
  std::cout<< "Drop data OK" << std::endl;
  fclose(pF);
  if( ptrprop.type!=cudaMemoryTypeHost ) delete datahost;
  return 0;
}

int dropChunk(){
//	return 0;
  Data* datahost = new Data;
  CHECK_ERROR( cudaMemcpy(datahost, parsHost.data, sizeof(Data), cudaMemcpyDeviceToHost) );
  char fname[256]; sprintf(fname, "Val_%07d.arr", parsHost.iStep);
  FILE* pF; pF = fopen(fname,"w");
  int sizeofT = sizeof(ftype);
  const int dim=3;
  int comsize = sizeof(double);
  double step = 1;
  const int scale=1;
  const int NxL = Nz/scale, NyL = Ny/scale, NzL=Nx/scale;
  fwrite(&comsize, sizeof(int  ), 1, pF);  //size of comment
  fwrite(&step   , comsize      , 1, pF);    // comment
  fwrite(&dim    , sizeof(int  ), 1, pF);     //dim =
  fwrite(&sizeofT, sizeof(int  ), 1, pF); //data size
  fwrite(&NxL     , sizeof(int  ), 1, pF);
  fwrite(&NyL     , sizeof(int  ), 1, pF);
  fwrite(&NzL     , sizeof(int  ), 1, pF);
  //printf("saving %s\n",fname);
  //for(int z=0; z<NzL; z++) for(int y=0; y<NyL; y++) for(int x=0; x<NxL; x++) {
  for(int x=0; x<NxL; x++) {
    const int3 crd = make_int3(x,0,0)*scale;
    //ftype val = datahost->tiles[crd.x].val[crd.y][crd.z];
    ftype* valp = 0;//datahost->tiles[crd.x].val.elem[0];
    //fwrite(valp, sizeof(ftype), NzL*NyL, pF);
    for(int iz=0;iz<Nz;iz++) fwrite(&valp[iz], sizeof(ftype), NzL*NyL, pF);
    printf(" Drop data ... progress = %2d%%       \r", (x+1)*100/NxL); fflush(stdout);
  }
  std::cout<< "Drop data OK" << std::endl;
  fclose(pF);
  delete datahost;
  return 0;
}

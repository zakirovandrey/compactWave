#pragma once
#include <cuda.h>
#include <assert.h>
#include <array>
#define USE_FLOAT
//#define USE_DOUBLE
#include "params.hpp"

const int PERFORMANCE_DEBUG=0;

#ifndef NGSIZE
#define NGSIZE 256
#endif

#ifndef NXSIZE
#define NXSIZE NGSIZE
#endif
#ifndef NYSIZE
#define NYSIZE NGSIZE
#endif
#ifndef NZSIZE
#define NZSIZE NGSIZE
#endif

const int Ncomp=2;

const int Nx=NXSIZE;//NTS*16;//120;
const int Ny=NYSIZE;//NTS*16;//120;
const int Nz=NZSIZE;//NTS*16;//120;

//static_assert(Nx*Ny*Nz<=INT_MAX);

#define MAX(a,b) ((a)>(b)?(a):(b))

template<const int NC> struct carray{
	union {
	    ftype elem[NC];
		ftype2 elem2;
	};
};

struct Tile {
  carray<Ncomp> val[Ny][Nz];
};

const int Nt = NT;//64;//NTS*8;

const ftype dt= 2*M_PI/0.999/320./2;//0.3;
const ftype dx=1.0;

const ftype dt2=dt*dt;

struct Data{
  Tile tiles[Nx];
};

template<int Ns> struct CellLine{
  ftype2 valflux[Ns][Nz];
};
struct Farsh{
  static_assert(Ny%2==0);
  static const int NLOAD=1;
  static const int NSAVE=1;
  static const int NFL = NT+1+NLOAD+NSAVE;
};
template<const int _NFY> struct FarshLines: public Farsh {
	static const int NFY=_NFY;
	CellLine<NFL> cls[NFY];
};
struct Cell{
  ftype val;
  ftype rhs;
};

struct Params{
  Data* data;
  Farsh* farsh[8];
  size_t farshsize;
  int Ngpus;
  int iStep;
  static const int Nt=NT;
} parsHost;

__constant__ Params pars;

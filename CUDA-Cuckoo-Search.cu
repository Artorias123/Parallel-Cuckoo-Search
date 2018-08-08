#include<iostream>
#include "stdlib.h"
#include <ctime>
#include <fstream>
#include <string>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
using namespace std;
#define pi 3.141592653
#define lamd 1.5
#define alpha 1
#define af 0.5
#define DIM 10
#define popNum 800
#define itNum 60000
__device__ float gpu_sigma,gpu_gar0;
float garmma(float x)
{
	float temp[2], d = 1;
	temp[0] = 1 / (x + 1);
	unsigned i = 2;
	for (; d > powf(0.1, 8); i++)
	{
		temp[1] = temp[0] * i / (x + i);
		d = temp[0] * powf(i - 1, x) - temp[1] * powf(i, x);
		temp[0] = temp[1];
	}
	return temp[0] * powf(i, x);
}
float gar0 = garmma(lamd),gar1 = garmma((1 + lamd) / 2),
	sigma = pow(gar0*sin(pi*lamd / 2) / (gar1*pow(2, (lamd - 1) / 2)), 1 / 2 * lamd);
//-------1---------2---------3---------4---------5---------6---------7---------8---------9---------
class nest
{
public:
	float *x,quality;
	unsigned dim,seed;
	nest(unsigned dim,float b=0,float e=1);
	~nest(){if(!x) free(x);}
	__host__ __device__ void get_quality();
	__host__ __device__ nest& operator=(const nest &y);
	__host__ __device__ float& operator[](const unsigned &i);
	__host__ __device__ const float& operator[](const unsigned &i) const;
	__device__ void levy_fly();
	__device__ int curand();
	__device__ float Gaussian();
};
__device__ int nest::curand() {
    seed = seed * 1103515245 + 12345;
    return((unsigned)(seed/65536) % 32768);
}

__device__ float nest::Gaussian()
{
	float a = float(curand()) / RAND_MAX, b = float(curand()) / RAND_MAX;
	return (cos(2 * pi*b))*powf(-2 * log(a), 0.5);
}
__host__ __device__ float f(const nest &x)
{
	float y=0,l=1;
	for (unsigned i = 0; i < x.dim; i++)
	{
		y += (x[i] - 100)*(x[i] - 100);
		l *= cos((x[i] - 100) / powf(i+1, 0.5));
	}
	y = y/4000-l+1;
	return y;
}
__device__ float g(const float *x,const unsigned &dim)
{
	float y=0,l=1;
	for (unsigned i = 0; i < dim; i++)
	{
		y += (x[i] - 100)*(x[i] - 100);
		l *= cos((x[i] - 100) / powf(i+1, 0.5));
	}
	y = y/4000-l+1;
	return y;
}
nest::nest(unsigned idim,float b,float e){
	dim=idim;
	quality=0;
	seed=rand();
	x=(float*)malloc(sizeof(float)*dim);
	for(unsigned i=0;i<dim;i++) x[i] = b + (e - b)*float(rand()) / RAND_MAX;
}
__host__ __device__ void nest::get_quality(){
	quality = 1 / (f(*this) + 0.00001);
}
__host__ __device__ float& nest::operator[](const unsigned &i){
	return x[i];
}
__host__ __device__ const float& nest::operator[](const unsigned &i) const{
	return x[i];
}
__host__ __device__ nest& nest::operator=(const nest &y){
    if(this!=&y){
		if(dim!=y.dim) {
			if(!x) free(x);
			x=(float*)malloc(sizeof(float)*y.dim);
		}
		dim=y.dim;
        for(unsigned i=0;i<dim;i++) x[i]=y.x[i];
		quality=y.quality;
    }
    return *this;
}
__device__ void nest::levy_fly(){
	float y[DIM], U, V, s, L;
	for (unsigned i = 0; i < dim; i++)
	{
		U = gpu_sigma*Gaussian();
		V = Gaussian();
		s = abs(U) / powf(abs(V), 1 / lamd);
		L = lamd*gpu_gar0*sin(pi*lamd / 2) / (pi*powf(s, 1 + lamd));
		y[i] = x[i] + alpha*L;
	}
	if (1 / (g(y,dim) + 0.00001)>quality)
	{
		for(unsigned i=0;i<dim;i++) x[i]=y[i];
		get_quality();
	}
}
//-------1---------2---------3---------4---------5---------6---------7---------8---------9---------
__device__ void elim(nest *pop,float *x,const unsigned &i){
	unsigned k[2];
	k[0] = pop[i].curand() % popNum;
	k[1] = pop[i].curand() % popNum;
	for (unsigned j = 0; j < DIM; j++) x[j] = pop[i][j] + af*(pop[k[0]][j] - pop[k[1]][j]);
	if (1 / (g(x,DIM) + 0.00001)>pop[i].quality)
	{
		for(unsigned j=0;j<DIM;j++) pop[i][j]=x[j];
		pop[i].get_quality();
	}
}
//-------1---------2---------3---------4---------5---------6---------7---------8---------9---------
__global__ void getq(nest *pop){
	unsigned i = threadIdx.x;
	pop[i].get_quality();
	float temp[DIM];
	for(unsigned j=0;j<itNum;j++){
		pop[i].levy_fly();
		elim(pop,temp,i);
	}
}
//-------1---------2---------3---------4---------5---------6---------7---------8---------9---------
int main()
{
	srand(unsigned(clock()));
	unsigned pop_num = popNum, dim = DIM;
	float a = -600, b = 600,**tmp;
	nest *pop, *gpu_pop;
	pop = (nest*)malloc(sizeof(nest)*pop_num);
	tmp = (float**)malloc(sizeof(float*)*pop_num);
	for (int i = 0; i<pop_num; i++) pop[i] = nest(dim, a, b);
//-------1---------2---------3---------4---------5---------6---------7---------8---------9---------
	cudaMemcpyToSymbol(gpu_gar0,&gar0,sizeof(float));
	cudaMemcpyToSymbol(gpu_sigma,&sigma,sizeof(float));
	cudaMalloc(&gpu_pop, sizeof(nest)*pop_num);
	for (unsigned i = 0; i < pop_num; i++) {
		cudaMalloc(&(tmp[i]), sizeof(float)*dim);
		cudaMemcpy(&(gpu_pop[i].x), &(tmp[i]), sizeof(float*), cudaMemcpyHostToDevice);
		cudaMemcpy(tmp[i], pop[i].x, sizeof(float)*dim, cudaMemcpyHostToDevice);
		cudaMemcpy(&(gpu_pop[i].dim), &(pop[i].dim), sizeof(unsigned), cudaMemcpyHostToDevice);
		cudaMemcpy(&(gpu_pop[i].seed), &(pop[i].seed), sizeof(unsigned), cudaMemcpyHostToDevice);
	}
	getq << <1, pop_num >> >(gpu_pop);
	getq << <1, pop_num >> >(gpu_pop);
	getq << <1, pop_num >> >(gpu_pop);
	getq << <1, pop_num >> >(gpu_pop);
	getq << <1, pop_num >> >(gpu_pop);
	unsigned maxn=0;
	cudaMemcpy(&(pop[0].quality), &(gpu_pop[0].quality), sizeof(float), cudaMemcpyDeviceToHost);
	float max=pop[0].quality;
	for (unsigned i = 1; i < pop_num; i++){
		cudaMemcpy(&(pop[i].quality), &(gpu_pop[i].quality), sizeof(float), cudaMemcpyDeviceToHost);
		if(pop[i].quality>max) {
			maxn=i;
			max=pop[i].quality;
		}
	}
	cudaMemcpy(pop[0].x, tmp[maxn], sizeof(float)*dim, cudaMemcpyDeviceToHost);
	pop[0].quality=max;
	for(unsigned i=0;i<DIM;i++)cout << "x" << i + 1 << "=" << pop[0][i] << '\t';
	cout <<endl<< "y=" <<f(pop[0])<< endl;
	cudaFree(gpu_pop);
	for(unsigned i=0;i<popNum;i++) cudaFree(tmp[i]);
	free(pop);
	free(tmp);
	return 0;
}

#include<iostream>
#include<math.h>
#include<vector>
#include "stdio.h"
#include "stdlib.h"
#include <ctime>
#include <fstream>
#include <string>
#include <algorithm>
#include "omp.h"
using namespace std;
#define pi 3.141592653
#define lamd 1.5
#define alpha 1
#define af 0.5
unsigned myrand(unsigned &seed){
	seed = seed * 1103515245 + 12345;
    return((unsigned)(seed/65536) % 32768);
}
double f(vector<double> x)
{
	double y=0,l=1;
	for (unsigned i = 0; i < x.size(); i++)
	{
		y += (x[i] - 100)*(x[i] - 100);
		l *= cos((x[i] - 100) / pow(i+1, 0.5));
	}
	y = y/4000-l+1;
	return y;
}
double H(double x, double t)
{
	if (x > t) return 1;
	else return 0;
}
double Gaussian(unsigned &seed)
{
	double a = double(myrand(seed)) / RAND_MAX, b = double(myrand(seed)) / RAND_MAX;
	return (cos(2 * pi*b))*pow(-2 * log(a), 0.5);
}
double garmma(double x)
{
	double temp[2], d = 1;
	temp[0] = 1 / (x + 1);
	unsigned i = 2;
	for (; d > pow(0.1, 8); i++)
	{
		temp[1] = temp[0] * i / (x + i);
		d = temp[0] * pow(i - 1, x) - temp[1] * pow(i, x);
		temp[0] = temp[1];
	}
	return temp[0] * pow(i, x);
}
class nest
{
public:
	vector<double> x;
	double quality,gar0,gar1;
	nest(){};
	nest(unsigned dim){ x.resize(dim, 0); gar0 = garmma(lamd); gar1 = garmma((1 + lamd) / 2); };
	void get_quality();
	void levy_fly(unsigned &seed);
	double similarity(nest &b);
	nest& operator =(const nest &b);
};
nest& nest::operator = (const nest &b){
	if(this!=&b){
		x=b.x;
		quality=b.quality;
		gar0=b.gar0;
		gar1=b.gar1;
	}
	return *this;
}
void nest::get_quality()
{
	quality = 1 / (f(x) + 0.00001);
}
void nest::levy_fly(unsigned &seed)
{
	vector<double> y(x.size());
	double sigma = pow(gar0*sin(pi*lamd / 2) / (gar1*pow(2, (lamd - 1) / 2)), 1 / 2 * lamd), U, V, s, L;
	for (unsigned i = 0; i < x.size(); i++)
	{
		U = sigma*Gaussian(seed);
		V = Gaussian(seed);
		s = abs(U) / pow(abs(V), 1 / lamd);
		L = lamd*gar0*sin(pi*lamd / 2) / (pi*pow(s, 1 + lamd));
		y[i] = x[i] + alpha*L;
	}
	if (1 / (f(y) + 0.00001)>quality)
	{
		x = y;
		get_quality();
	}
}
bool lessn(const nest &a,const nest &b){
	if(a.quality<b.quality) return 0;
	else return 1;
}
class population
{
public:
	vector<nest> p;
	double e, b;
	population(double a, double c, unsigned n, unsigned dim){ b = a; e = c; p.resize(n, nest(dim)); init(0); get_quality(0,p.size()); };
	void init(unsigned h);
	void get_quality(unsigned begin,unsigned end);
	void elim(unsigned begin,unsigned end,unsigned &seed);
	void find(unsigned begin,unsigned end,unsigned &seed);
};
void population::init(unsigned h)
{
	srand(h);
	for (unsigned i = 0; i < p.size(); i++)
	{
		for (unsigned j = 0; j < p[0].x.size(); j++)
		{
			p[i].x[j] = b + (e - b)*double(rand()) / RAND_MAX;
		}
	}
}
void population::get_quality(unsigned begin,unsigned end)
{
	for (unsigned i = begin; i < end; i++) p[i].get_quality();
}
void population::elim(unsigned begin,unsigned end,unsigned &seed)
{
	vector<double> temp(p[0].x.size());
	for (unsigned i = begin; i < end; i++)
	{
		int k[2];
		k[0] = myrand(seed) % p.size();
		k[1] = myrand(seed) % p.size();
		for (unsigned j = 0; j < p[0].x.size(); j++){ 
			temp[j] = p[i].x[j] + af*(p[k[0]].x[j] - p[k[1]].x[j]);
		}
		if (p[i].quality < 1 / (0.00001 + f(temp))) p[i].x = temp;
		p[i].get_quality();
	}
}
void population::find(unsigned begin,unsigned end,unsigned &seed)
{
	for (unsigned i = begin; i < end; i++)
	{
		p[i].levy_fly(seed);
	}
}
class calculator
{
public:
	unsigned n;
	double begin, end;
	population *p;
	nest best;
	calculator(population &x, unsigned in, double a, double b){ begin = a; end = b; n = in; p = &x; best = p->p[0]; };
	void advance(unsigned b,unsigned e,unsigned &seed);
	void iteration(ofstream &fout);
};
void calculator::advance(unsigned b,unsigned e,unsigned &seed)
{
	p->elim(b,e,seed);
	p->find(b,e,seed);
}
void calculator::iteration(ofstream &fout)
{
	int nt=4;
	omp_set_num_threads(nt);
	#pragma omp parallel
	{
		int d=p->p.size()/nt,a=omp_get_thread_num()*d,b=a+d;
		//printf("id = %d,start = %d,end = %d\n",omp_get_thread_num(),a,b);
		//double start_time = omp_get_wtime();
		unsigned seed=rand();
		for (unsigned i = 0; i < n; i++)
		{
			if (i % 10000 == 0) srand(rand() % unsigned(clock()));
			/*if (i == 0)
			{
				for (unsigned j = 0; j < p->p[i].x.size();j++)cout << p->p[i].x[j] << '\t';
				cout << endl;
			}*/
			//double start_ti = omp_get_wtime();
			advance(a,b,seed);
			//double end_ti = omp_get_wtime();
			//printf("step = %d,id = %d,time = %f\n",i,omp_get_thread_num(),(end_ti - start_ti));
		}
		double end_time = omp_get_wtime();
		//printf("id = %d,time = %f\n",omp_get_thread_num(),(end_time - start_time));
	}
	best=*min_element(p->p.begin(),p->p.end(),lessn);
	for (unsigned i = 0; i < best.x.size(); i++)
	{
		cout << "x" << i + 1 << "=" << best.x[i] << '\t';
	}
	cout << endl << "y=" << f(best.x) << endl;
	for (unsigned i = 0; i < best.x.size(); i++)
	{
		fout << best.x[i] << '\t';
	}
	fout << endl;
}
int main()
{
	//clock_t start_time = clock();
	srand(unsigned(clock()));
	ofstream fout("result", ios::out | ios::trunc);
	fout.precision(12);
	unsigned pop_num = 400, it_num = 80000, dim = 9;
	double a = -600, b = 600;
	srand(unsigned(clock()));
	population p(a, b, pop_num, dim);
	calculator C(p, it_num, a, b);
	//for (unsigned i = 0; i < 100; i++)
	//{
		C.p->init(rand());
		C.iteration(fout);
	//}	
	fout.close();
	//clock_t end_time = clock();
	//cout << "time=" << static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC << "s" << endl;
	return 0;
}
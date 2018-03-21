#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <limits>
#include <vector>
#include <random>
#include <cmath>
#include <complex>
#include <numeric>
#include <boost/numeric/odeint.hpp>
#include <omp.h>
#include <boost/numeric/odeint/external/openmp/openmp.hpp>
#include "param.h"
/*
namespace MODEL
{
  const int N_GRIDS_1D = 210/2;
  const int N_GRIDS_3D = N_GRIDS_1D*N_GRIDS_1D*N_GRIDS_1D;
  const int N_VARIABLES = 4+N_GRIDS_1D*N_GRIDS_1D*N_GRIDS_1D*6;
  const double X_MAX = 105/2;
  const double X_MIN = -105/2;
  // Parameters to control Integrate ODS precision
  const double dt = 0.05;
  const double t_max = 102;
  const size_t n_print = 20;
  // Parameters to control the absorption boundary condition
  const double X_CUTOFF = 28;
  const double X_MIDDLE = 16;
  // Initialize OMEGA
  const double OMEGA = 0.25;
  // Initialize strength of P
  const double P_MAX =  0.025; //0.025
  const double SIGMA = 0.50; //0.3
  // FGR rate in 3D
  const double GAMMA = P_MAX*P_MAX*OMEGA*OMEGA*OMEGA/3/3.1415926;

  // Parameters to control the place to store varibales
  // Some starting points;
  const size_t C1_START = 0;
  const size_t C2_START = 2;
  const size_t Ex_START = 4;
  const size_t Ey_START = Ex_START + N_GRIDS_3D;
  const size_t Ez_START = Ey_START + N_GRIDS_3D;
  const size_t Bx_START = Ez_START + N_GRIDS_3D;
  const size_t By_START = Bx_START + N_GRIDS_3D;
  const size_t Bz_START = By_START + N_GRIDS_3D;
}
*/
using namespace std;
using namespace boost::numeric::odeint;
using namespace MODEL;

const complex<double> I(0, 1);
const double PI = 3.1415926;

typedef std::vector<double> state_type;

// Decalriation of functions
double get_random_mt19937();

double get_random_01()
{
    //auto seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    //auto real_rand = std::bind(std::uniform_real_distribution<double>(0,1), mt19937(seed));
    //static std::mt19937 e;
    //static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
    //return real_rand();
    static std::mt19937 e;
    static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
    return dis(e);

}

double get_random_angle()
{
    return 2.0 * PI * get_random_01();
}

int get_plus_or_minus_one()
{
    if (get_random_01() > 0.5)
        return -1;
    else
        return 1;
}


// Define the class to perform ODEs
class ClassicalEM_2LevelSystem_3D
{
private:
    std::vector<double> Rx, Ry, Rz, Px, Py, Pz;
    double p_max, omega, dR;
	std::vector<double> TBx, TBy, TBz;
	std::vector<double> TEx, TEy, TEz;
    double intTE2, intTB2;
    double KAPPA_B, KAPPA_E;
public:
    ClassicalEM_2LevelSystem_3D(const double x_max=35, const double x_min=-35,
        const double Omega=0.25, const double P_max=0.025, const double sigma=1.0):omega(Omega), p_max(P_max)
        {
            Rx.resize(N_GRIDS_3D);
            Ry.resize(N_GRIDS_3D);
            Rz.resize(N_GRIDS_3D);
            Px.resize(N_GRIDS_3D);
            Py.resize(N_GRIDS_3D);
            Pz.resize(N_GRIDS_3D);

			// Transition directions
            TEx.resize(N_GRIDS_3D);
            TEy.resize(N_GRIDS_3D);
            TEz.resize(N_GRIDS_3D);
            TBx.resize(N_GRIDS_3D);
            TBy.resize(N_GRIDS_3D);
            TBz.resize(N_GRIDS_3D);

            dR = (x_max - x_min)/(N_GRIDS_1D - 1);
            #pragma omp parallel for schedule(runtime)
            for(size_t t=0; t < N_GRIDS_3D; t++)
            {
                size_t t1 = t % ( N_GRIDS_1D * N_GRIDS_1D );
                size_t k = t1 % N_GRIDS_1D;
                size_t j = (t1 - k) / N_GRIDS_1D;
                size_t i = (t - t1) / (N_GRIDS_1D*N_GRIDS_1D);

                size_t ijk = i*N_GRIDS_1D*N_GRIDS_1D + j*N_GRIDS_1D + k;
                Rx[ijk] = x_min + dR*i;
                Ry[ijk] = x_min + dR*j;
                Rz[ijk] = x_min + dR*k;
                // This is the gaussian shape
                double coeff = (2.0*sigma*sigma*sqrt(sigma)/PI/sqrt(PI))*p_max
                               *exp(-sigma*(Rx[ijk]*Rx[ijk] + Ry[ijk]*Ry[ijk] + Rz[ijk]*Rz[ijk]));

                //Px[ijk] = coeff*Rz[ijk]*Rx[ijk];
                //Py[ijk] = coeff*Rz[ijk]*Ry[ijk];
                //Pz[ijk] = coeff*Rz[ijk]*Rz[ijk];
                Px[ijk] = 0.0;
                Py[ijk] = 0.0;
                Pz[ijk] = coeff;

                //// TE = P
                //TEx[ijk] = Px[ijk];
                //TEy[ijk] = Py[ijk];
                //TEz[ijk] = Pz[ijk];

                //TE = del X del X P - g P
                TEx[ijk] = 4.0*sigma*sigma*coeff*Rz[ijk]*Rx[ijk];
                TEy[ijk] = 4.0*sigma*sigma*coeff*Rz[ijk]*Ry[ijk];
                TEz[ijk] =-4.0*sigma*sigma*coeff*( Rx[ijk]*Rx[ijk]+Ry[ijk]*Ry[ijk] );

                //TB = del X P
                TBx[ijk] = -2.0*sigma*coeff*Ry[ijk];
                TBy[ijk] =  2.0*sigma*coeff*Rx[ijk];
                TBz[ijk] =  0.0;

                //TE = far-field radiation 
                //TEx[ijk] =-sigma*sigma*coeff*Rz[ijk]*Rx[ijk];
                //TEy[ijk] =-sigma*sigma*coeff*Rz[ijk]*Ry[ijk];
                //TEz[ijk] = sigma*sigma*coeff*( Rx[ijk]*Rx[ijk]+Ry[ijk]*Ry[ijk] );
                
                //TB = far-field radiation
                //TBx[ijk] = sigma*coeff*Ry[ijk];
                //TBy[ijk] =-sigma*coeff*Rx[ijk];
                //TBz[ijk] = 0.0;


                // This is the hard sphere
                /*
                if(Rx[ijk]*Rx[ijk] + Ry[ijk]*Ry[ijk] + Rz[ijk]*Rz[ijk] <= 1.0/2.0/sigma)
                {
                    double coeff = 3.0 *2.0*sigma*sqrt(2.0*sigma)/4.0/PI*p_max;
                    Px[ijk] = 0.0;
                    Py[ijk] = 0.0;
                    Pz[ijk] = coeff;
                }
                else
                {
                    Px[ijk] = 0.0;
                    Py[ijk] = 0.0;
                    Pz[ijk] = 0.0;
                }
                */
            }

            // Calculate intTE2 and intTB2
            intTE2 = 0.0;
            //#pragma omp parallel for reduction(+:intPP) schedule(runtime)
            for(size_t tt=0; tt<N_GRIDS_3D; tt++)
            {
                intTE2 += TEx[tt]*TEx[tt] + TEy[tt]*TEy[tt] + TEz[tt]*TEz[tt];
            }
            intTE2 *= dR*dR*dR;

            intTB2 = 0.0;
            //#pragma omp parallel for reduction(+:intTT) schedule(runtime)
            for(size_t tt=0; tt<N_GRIDS_3D; tt++)
            {
                intTB2 += TBx[tt]*TBx[tt] + TBy[tt]*TBy[tt] + TBz[tt]*TBz[tt];
            }
            intTB2 *= dR*dR*dR;

        }
    double get_dR()
    {
        return dR;
    }
    double get_r_norm(size_t ith_in_N_3D)
    {
        return sqrt(Rx[ith_in_N_3D]*Rx[ith_in_N_3D] + Ry[ith_in_N_3D]*Ry[ith_in_N_3D] + Rz[ith_in_N_3D]*Rz[ith_in_N_3D]);
    }

    void operator() (const state_type &x, state_type &dxdt, const double /* t */)
    {
        /* Step 1. Initialize Elements of A matrix: Hs - Integral{drE(r)P(r)} */
        double A_11 = 0.0, A_22 = omega;
        //double A_12 = -dR*dR*dR*(std::inner_product(Px.begin(), Px.end(), x.begin()+Ex_START, 0.0) +
        //    std::inner_product(Py.begin(), Py.end(), x.begin()+Ey_START, 0.0) + std::inner_product(Pz.begin(), Pz.end(), x.begin()+Ez_START, 0.0) );
        // Need to change this code to parallel version
        double A_12 = 0.0;
        #pragma omp parallel for reduction(+:A_12) schedule(runtime)
        for(size_t tt=0; tt<N_GRIDS_3D; tt++)
        {
            A_12 += Px[tt]*x[tt+Ex_START] + Py[tt]*x[tt+Ey_START] + Pz[tt]*x[tt+Ez_START];
        }
        A_12 *= -dR*dR*dR;
        double A_21 = A_12;
        /* Step 2. Calculate the complex amplitue C1 and C2 */
        /* Compute C1: Re(C1) has location 0, Im(C1) has location 1
		Re(dC1dt) = A_12*Im(C2) + A_11*Im(C1); Im(dC1dt) = -A_12*Re(C2) - A_11*Re(C1) */
		dxdt[0] = A_12*x[3] + A_11*x[1];
		dxdt[1] = -A_12*x[2] - A_11*x[0];
		/* Compute C2: Re(C2) has location 2, Im(C1) has location 3
		Re(dC2dt) = A_21*Im(C1) + A_22*Im(C2); Im(dC2dt) = -A_21*Re(C1) - A_22*Re(C2) */
		dxdt[2] = A_21*x[1] + A_22*x[3];
		dxdt[3] = -A_21*x[0] - A_22*x[2];
        /* Step 3. Calculate the displacement current Jx, Jy, Jz */
        /* Compute Jx, Jy and Jz: Jx(r) = - 2A_22*Im(C1*C2')*Px(r), Jy(r) = - 2A_22*Im(C1*C2')*Py(r), Jz(r) = -2A_22*Im(C1*C2')*Pz(r)*/

		double coefficient = -2.0*A_22*(complex<double>(x[0], x[1])*complex<double>(x[2], -x[3])).imag();
        //coefficient = 0.0; // Set the current to be zero

		#pragma omp parallel for schedule(runtime)
		for (size_t t = 0; t < N_GRIDS_3D; t++)
		{
       		double Jx = coefficient*Px[t];
			double Jy = coefficient*Py[t];
			double Jz = coefficient*Pz[t];
            /* Step 4. Calculate time derivative of Ex, Ey, Ez, Bx, By, Bz*/
            // Here we do not need to worry about EM at boundary, because smoothing function will let them 0
            size_t t1 = t % ( N_GRIDS_1D * N_GRIDS_1D );
            size_t k = t1 % N_GRIDS_1D;
            size_t j = (t1 - k) / N_GRIDS_1D;
            size_t i = (t - t1) / (N_GRIDS_1D*N_GRIDS_1D);

            size_t ijk = i*N_GRIDS_1D*N_GRIDS_1D + j*N_GRIDS_1D + k;

            size_t i_plus_jk = (i+1)*N_GRIDS_1D*N_GRIDS_1D + j*N_GRIDS_1D + k;
            size_t i_minus_jk = (i-1)*N_GRIDS_1D*N_GRIDS_1D + j*N_GRIDS_1D + k;
            size_t ij_plus_k = i*N_GRIDS_1D*N_GRIDS_1D + (j+1)*N_GRIDS_1D + k;
            size_t ij_minus_k = i*N_GRIDS_1D*N_GRIDS_1D + (j-1)*N_GRIDS_1D + k;
            size_t ijk_plus = i*N_GRIDS_1D*N_GRIDS_1D + j*N_GRIDS_1D + k+1;
            size_t ijk_minus = i*N_GRIDS_1D*N_GRIDS_1D + j*N_GRIDS_1D + k-1;

            size_t i_plus2_jk = (i+2)*N_GRIDS_1D*N_GRIDS_1D + j*N_GRIDS_1D + k;
            size_t i_minus2_jk = (i-2)*N_GRIDS_1D*N_GRIDS_1D + j*N_GRIDS_1D + k;
            size_t ij_plus2_k = i*N_GRIDS_1D*N_GRIDS_1D + (j+2)*N_GRIDS_1D + k;
            size_t ij_minus2_k = i*N_GRIDS_1D*N_GRIDS_1D + (j-2)*N_GRIDS_1D + k;
            size_t ijk_plus2 = i*N_GRIDS_1D*N_GRIDS_1D + j*N_GRIDS_1D + k+2;
            size_t ijk_minus2 = i*N_GRIDS_1D*N_GRIDS_1D + j*N_GRIDS_1D + k-2;

            size_t i_plus3_jk = (i+3)*N_GRIDS_1D*N_GRIDS_1D + j*N_GRIDS_1D + k;
            size_t i_minus3_jk = (i-3)*N_GRIDS_1D*N_GRIDS_1D + j*N_GRIDS_1D + k;
            size_t ij_plus3_k = i*N_GRIDS_1D*N_GRIDS_1D + (j+3)*N_GRIDS_1D + k;
            size_t ij_minus3_k = i*N_GRIDS_1D*N_GRIDS_1D + (j-3)*N_GRIDS_1D + k;
            size_t ijk_plus3 = i*N_GRIDS_1D*N_GRIDS_1D + j*N_GRIDS_1D + k+3;
            size_t ijk_minus3 = i*N_GRIDS_1D*N_GRIDS_1D + j*N_GRIDS_1D + k-3;

            if( i==0 || i==N_GRIDS_1D-1 || j==0 || j==N_GRIDS_1D-1 || k==0 || k==N_GRIDS_1D-1)
                {
                    dxdt[Bx_START + ijk] =  0.0;
                    dxdt[By_START + ijk] =  0.0;
                    dxdt[Bz_START + ijk] =  0.0;
                    dxdt[Ex_START + ijk] =  0.0;
                    dxdt[Ey_START + ijk] =  0.0;
                    dxdt[Ez_START + ijk] =  0.0 ;
                }
            else if(i==1 || i==N_GRIDS_1D-2 || j==1 || j==N_GRIDS_1D-2 || k==1 || k==N_GRIDS_1D-2 )
            {
                dxdt[Bx_START + ijk] = -( (x[Ez_START + ij_plus_k] - x[Ez_START + ij_minus_k]) - (x[Ey_START + ijk_plus] - x[Ey_START + ijk_minus]) )/2.0/dR;
                dxdt[By_START + ijk] =  ( (x[Ez_START + i_plus_jk] - x[Ez_START + i_minus_jk]) - (x[Ex_START + ijk_plus] - x[Ex_START + ijk_minus]) )/2.0/dR;
                dxdt[Bz_START + ijk] = -( (x[Ey_START + i_plus_jk] - x[Ey_START + i_minus_jk]) - (x[Ex_START + ij_plus_k] - x[Ex_START + ij_minus_k]) )/2.0/dR;
                dxdt[Ex_START + ijk] =  ( (x[Bz_START + ij_plus_k] - x[Bz_START + ij_minus_k]) - (x[By_START + ijk_plus] - x[By_START + ijk_minus]) )/2.0/dR - Jx;
                dxdt[Ey_START + ijk] = -( (x[Bz_START + i_plus_jk] - x[Bz_START + i_minus_jk]) - (x[Bx_START + ijk_plus] - x[Bx_START + ijk_minus]) )/2.0/dR -Jy;
                dxdt[Ez_START + ijk] =  ( (x[By_START + i_plus_jk] - x[By_START + i_minus_jk]) - (x[Bx_START + ij_plus_k] - x[Bx_START + ij_minus_k]) )/2.0/dR -Jz;
            }
            else if(i==2 || i==N_GRIDS_1D-3 || j==2 || j==N_GRIDS_1D-3 || k==2 || k==N_GRIDS_1D-3 )
            {
                dxdt[Bx_START + ijk] = -( ( -x[Ez_START + ij_plus2_k] + 8.0*x[Ez_START + ij_plus_k] - 8.0*x[Ez_START + ij_minus_k] + x[Ez_START + ij_minus2_k])
                                        - ( -x[Ey_START + ijk_plus2]  + 8.0*x[Ey_START + ijk_plus]  - 8.0*x[Ey_START + ijk_minus]  + x[Ey_START + ijk_minus2]) )/12.0/dR;
                dxdt[By_START + ijk] =  ( ( -x[Ez_START + i_plus2_jk] + 8.0*x[Ez_START + i_plus_jk] - 8.0*x[Ez_START + i_minus_jk] + x[Ez_START + i_minus2_jk])
                                        - ( -x[Ex_START + ijk_plus2]  + 8.0*x[Ex_START + ijk_plus]  - 8.0*x[Ex_START + ijk_minus]  + x[Ex_START + ijk_minus2]) )/12.0/dR;
                dxdt[Bz_START + ijk] = -( ( -x[Ey_START + i_plus2_jk] + 8.0*x[Ey_START + i_plus_jk] - 8.0*x[Ey_START + i_minus_jk] + x[Ey_START + i_minus2_jk])
                                        - ( -x[Ex_START + ij_plus2_k] + 8.0*x[Ex_START + ij_plus_k] - 8.0*x[Ex_START + ij_minus_k] + x[Ex_START + ij_minus2_k]) )/12.0/dR;
                dxdt[Ex_START + ijk] = ( ( -x[Bz_START + ij_plus2_k] + 8.0*x[Bz_START + ij_plus_k] - 8.0*x[Bz_START + ij_minus_k] + x[Bz_START + ij_minus2_k])
                                        - ( -x[By_START + ijk_plus2]  + 8.0*x[By_START + ijk_plus]  - 8.0*x[By_START + ijk_minus]  + x[By_START + ijk_minus2]) )/12.0/dR -  Jx;
                dxdt[Ey_START + ijk] =  -( ( -x[Bz_START + i_plus2_jk] + 8.0*x[Bz_START + i_plus_jk] - 8.0*x[Bz_START + i_minus_jk] + x[Bz_START + i_minus2_jk])
                                        - ( -x[Bx_START + ijk_plus2]  + 8.0*x[Bx_START + ijk_plus]  - 8.0*x[Bx_START + ijk_minus]  + x[Bx_START + ijk_minus2]) )/12.0/dR -  Jy;
                dxdt[Ez_START + ijk] = ( ( -x[By_START + i_plus2_jk] + 8.0*x[By_START + i_plus_jk] - 8.0*x[By_START + i_minus_jk] + x[By_START + i_minus2_jk])
                                        - ( -x[Bx_START + ij_plus2_k] + 8.0*x[Bx_START + ij_plus_k] - 8.0*x[Bx_START + ij_minus_k] + x[Bx_START + ij_minus2_k]) )/12.0/dR - Jz;
            }
            else
            {
                dxdt[Bx_START + ijk] = -( ( x[Ez_START + ij_plus3_k] - 9.0*x[Ez_START + ij_plus2_k] + 45.0*x[Ez_START + ij_plus_k] - 45.0*x[Ez_START + ij_minus_k] + 9.0*x[Ez_START + ij_minus2_k] - x[Ez_START + ij_minus3_k])
                                        - ( x[Ey_START + ijk_plus3]  - 9.0*x[Ey_START + ijk_plus2]  + 45.0*x[Ey_START + ijk_plus]  - 45.0*x[Ey_START + ijk_minus]  + 9.0*x[Ey_START + ijk_minus2]  - x[Ey_START + ijk_minus3]) )/60.0/dR;
                dxdt[By_START + ijk] =  ( ( x[Ez_START + i_plus3_jk] - 9.0*x[Ez_START + i_plus2_jk] + 45.0*x[Ez_START + i_plus_jk] - 45.0*x[Ez_START + i_minus_jk] + 9.0*x[Ez_START + i_minus2_jk] - x[Ez_START + i_minus3_jk])
                                        - ( x[Ex_START + ijk_plus3]  - 9.0*x[Ex_START + ijk_plus2]  + 45.0*x[Ex_START + ijk_plus]  - 45.0*x[Ex_START + ijk_minus]  + 9.0*x[Ex_START + ijk_minus2]  - x[Ex_START + ijk_minus3]) )/60.0/dR;
                dxdt[Bz_START + ijk] = -( ( x[Ey_START + i_plus3_jk] - 9.0*x[Ey_START + i_plus2_jk] + 45.0*x[Ey_START + i_plus_jk] - 45.0*x[Ey_START + i_minus_jk] + 9.0*x[Ey_START + i_minus2_jk] - x[Ey_START + i_minus3_jk])
                                        - ( x[Ex_START + ij_plus3_k] - 9.0*x[Ex_START + ij_plus2_k] + 45.0*x[Ex_START + ij_plus_k] - 45.0*x[Ex_START + ij_minus_k] + 9.0*x[Ex_START + ij_minus2_k] - x[Ex_START + ij_minus3_k]) )/60.0/dR;
                dxdt[Ex_START + ijk] = ( (  x[Bz_START + ij_plus3_k] - 9.0*x[Bz_START + ij_plus2_k] + 45.0*x[Bz_START + ij_plus_k] - 45.0*x[Bz_START + ij_minus_k] + 9.0*x[Bz_START + ij_minus2_k] - x[Bz_START + ij_minus3_k])
                                        - ( x[By_START + ijk_plus3]  - 9.0*x[By_START + ijk_plus2]  + 45.0*x[By_START + ijk_plus]  - 45.0*x[By_START + ijk_minus]  + 9.0*x[By_START + ijk_minus2]  - x[By_START + ijk_minus3]) )/60.0/dR -  Jx;
                dxdt[Ey_START + ijk] = -( ( x[Bz_START + i_plus3_jk] - 9.0*x[Bz_START + i_plus2_jk] + 45.0*x[Bz_START + i_plus_jk] - 45.0*x[Bz_START + i_minus_jk] + 9.0*x[Bz_START + i_minus2_jk] - x[Bz_START + i_minus3_jk])
                                        - ( x[Bx_START + ijk_plus3]  - 9.0*x[Bx_START + ijk_plus2]  + 45.0*x[Bx_START + ijk_plus]  - 45.0*x[Bx_START + ijk_minus]  + 9.0*x[Bx_START + ijk_minus2]  - x[Bx_START + ijk_minus3]) )/60.0/dR -  Jy;
                dxdt[Ez_START + ijk] = ( (  x[By_START + i_plus3_jk] - 9.0*x[By_START + i_plus2_jk] + 45.0*x[By_START + i_plus_jk] - 45.0*x[By_START + i_minus_jk] + 9.0*x[By_START + i_minus2_jk] - x[By_START + i_minus3_jk])
                                        - ( x[Bx_START + ij_plus3_k] - 9.0*x[Bx_START + ij_plus2_k] + 45.0*x[Bx_START + ij_plus_k] - 45.0*x[Bx_START + ij_minus_k] + 9.0*x[Bx_START + ij_minus2_k] - x[Bx_START + ij_minus3_k]) )/60.0/dR - Jz;
            }

		}

    }

    void augment(state_type &x, const double t)
    {
        const double P1 = x[0]*x[0]+x[1]*x[1];
        const double P2 = x[2]*x[2]+x[3]*x[3];
        double k_R = GAMMA * P2;
        //const double k_R = GAMMA * P2 * 2.0* sin(OMEGA*t)*sin(OMEGA*t); 
        double ImRho12 = -x[0]*x[3] + x[1]*x[2];
        if (P1*P2 != 0.0)
        {
            k_R = k_R* 2.0*ImRho12*ImRho12/P1/P2;
        }
        //(2.0*abs( (rho(0,1)/abs(rho(0,1))).real() ) )
        double additional_population_loss = k_R * P2 * dt;
        double additional_energy_loss = OMEGA * additional_population_loss;
        if  ( abs(P2) < 1e-10 ) // ground state do nothing
        {
            ;
        }
        else if ( abs(P2 - 1.0) < 1e-10 ) // excited state do something
        {
          //  cout << "Initial" << endl;
            x[0] += sqrt(additional_population_loss);
            x[2] = x[2]*sqrt(1.0-additional_population_loss);
            x[3] = x[3]*sqrt(1.0-additional_population_loss);
        }
        else if ( P2 - additional_population_loss > 0)
        //excited state population is big enough to decay
        {
            x[0] = x[0]*sqrt((P1+additional_population_loss)/P1);
            x[1] = x[1]*sqrt((P1+additional_population_loss)/P1);
            x[2] = x[2]*sqrt((P2-additional_population_loss)/P2);
            x[3] = x[3]*sqrt((P2-additional_population_loss)/P2);
        }

        // Add the additional energy loss to EM field randomly
        double theta = get_random_angle();
        //double dU_E = sin(theta) * sin(theta) * additional_energy_loss;
        //double dU_B = cos(theta) * cos(theta) * additional_energy_loss;
        //cout << "P2 = " << P2 << endl;
        //cout << "additional_energy_loss= " << additional_energy_loss<<" dU_E = " << dU_E << "; dU_B = " << dU_B << endl;
        //cout << dU_E << dU_B << endl;
        double dU_E = 0.5*additional_energy_loss;
        double dU_B = 0.5*additional_energy_loss;
        double kappa_E=0.0;
        double kappa_B=0.0;

        // Prepare the integrals
        double intETE = 0.0;
        #pragma omp parallel for reduction(+:intETE) schedule(runtime)
        for(size_t tt=0; tt<N_GRIDS_3D; tt++)
        {
            intETE += TEx[tt]*x[tt+Ex_START] + TEy[tt]*x[tt+Ey_START] + TEz[tt]*x[tt+Ez_START];
        }
        intETE *= dR*dR*dR;

        double intBTB = 0.0;
        #pragma omp parallel for reduction(+:intBTB) schedule(runtime)
        for(size_t tt=0; tt<N_GRIDS_3D; tt++)
        {
            intBTB += TBx[tt]*x[tt+Bx_START] + TBy[tt]*x[tt+By_START] + TBz[tt]*x[tt+Bz_START];
        }
        intBTB *= dR*dR*dR;

        /*
        // NEW RESCALE
        double dU = 2*additional_energy_loss + intETE*intETE/intTE2 + intBTB*intBTB/intTB2;
        double alpha = -intETE/intTE2 + sqrt(dU/intTE2) * cos(theta);
        double beta  = -intBTB/intTB2 + sqrt(dU/intTB2) * sin(theta);
        while (alpha*beta <0.0)
        {
            theta = get_random_angle();
            alpha = -intETE/intTE2 + sqrt(dU/intTE2) * cos(theta);
            beta  = -intBTB/intTB2 + sqrt(dU/intTB2) * sin(theta);
        }

        #pragma omp parallel for schedule(runtime)
        for(size_t tt=0; tt < N_GRIDS_3D; tt++)
        {
            x[Ex_START+tt] += alpha * TEx[tt];
            x[Ey_START+tt] += alpha * TEy[tt];
            x[Ez_START+tt] += alpha * TEz[tt];
            x[Bx_START+tt] += beta  * TBx[tt];
            x[By_START+tt] += beta  * TBy[tt];
            x[Bz_START+tt] += beta  * TBz[tt];
        }
        */
        /*
        // TEST: energy conservation
        double energy_E = 0.0;
        #pragma omp parallel for reduction(+:energy_E) schedule(runtime)
        for(size_t tt=0; tt<N_GRIDS_3D; tt++)
        {
            energy_E += x[tt+Ex_START]*x[tt+Ex_START] + x[tt+Ey_START]*x[tt+Ey_START] + x[tt+Ez_START]*x[tt+Ez_START];
        }
        energy_E *= dR*dR*dR/2;
        // TEST: energy conservation
        double energy_B = 0.0;
        #pragma omp parallel for reduction(+:energy_B) schedule(runtime)
        for(size_t tt=0; tt<N_GRIDS_3D; tt++)
        {
            energy_B += x[tt+Bx_START]*x[tt+Bx_START] + x[tt+By_START]*x[tt+By_START] + x[tt+Bz_START]*x[tt+Bz_START];
        }
        energy_B *= dR*dR*dR/2;
        */

        
        // calculate coefficients and add to EM field
        //double de_check = 0.0;
        //double dh_check = 0.0;
        if ( abs(intETE) == 0.0 )
        {
            //int random_number = get_plus_or_minus_one();
            int random_number = 1;
            //Ez += (random_number * Pz * sqrt(2.0 * de_Ez / intPP) );
            #pragma omp parallel for schedule(runtime)
            for(size_t tt=0; tt < N_GRIDS_3D; tt++)
            {
                x[Ex_START+tt] += random_number * TEx[tt] * sqrt(2.0 * dU_E / intTE2);
                x[Ey_START+tt] += random_number * TEy[tt] * sqrt(2.0 * dU_E / intTE2);
                x[Ez_START+tt] += random_number * TEz[tt] * sqrt(2.0 * dU_E / intTE2);
                //x[Ex_START+tt] = TEx[tt];
                //x[Ey_START+tt] = TEy[tt];
                //x[Ez_START+tt] = TEz[tt];
            }
            //cout << "first energy: " << sqrt(2.0 * dU_E / intTE2) << endl;
           // de_check = 0.50 * dx * arma::accu(Ez % Ez) - de_Ez;
           // cout << "De_CHECK_Ez = " << de_check << endl;
        }
        else
        {
            double root1_E = (-intETE + sqrt(intETE*intETE + 2.0 * dU_E * intTE2) ) / intTE2;
            double root2_E = (-intETE - sqrt(intETE*intETE + 2.0 * dU_E * intTE2) ) / intTE2;
            //if( abs(root1_E) < abs(root2_E))
            if (abs(root1_E + KAPPA_E)<abs(root2_E + KAPPA_E))
                //if (root1_E > 0.0)
                kappa_E = root1_E;
            else
                kappa_E = root2_E;
            #pragma omp parallel for schedule(runtime)
            for(size_t tt=0; tt < N_GRIDS_3D; tt++)
            {
                x[Ex_START+tt] += kappa_E * TEx[tt];
                x[Ey_START+tt] += kappa_E * TEy[tt];
                x[Ez_START+tt] += kappa_E * TEz[tt];
            }
            KAPPA_E += kappa_E;

            //double energy_ER = 0.0;
            //#pragma omp parallel for reduction(+:energy_ER) schedule(runtime)
            //for(size_t tt=0; tt<N_GRIDS_3D; tt++)
            //{
            //energy_ER += x[tt+Ex_START]*x[tt+Ex_START] + x[tt+Ey_START]*x[tt+Ey_START] + x[tt+Ez_START]*x[tt+Ez_START];
            //}
            //energy_ER *= dR*dR*dR/2;
            //
            ////de_check = 0.50 * dx * kappa_Ez * kappa_Ez * arma::accu(Pz % Pz) - de_Ez;
            //cout << "DU_CHECK_E = " << energy_ER-energy_E << "; dU_E = " << dU_E << endl;
        }

        if ( abs(intBTB) == 0.0 )
        {
            //int random_number = get_plus_or_minus_one() ;
            int random_number = 1;
            //Hy += (random_number1 * dPzdx * sqrt( 2.0 * de_Hy / intdPzdxdPzdx)) ;
            #pragma omp parallel for schedule(runtime)
            for(size_t tt=0; tt < N_GRIDS_3D; tt++)
            {
                x[Bx_START+tt] += random_number * TBx[tt] * sqrt(2.0 * dU_B / intTB2);
                x[By_START+tt] += random_number * TBy[tt] * sqrt(2.0 * dU_B / intTB2);
                x[Bz_START+tt] += random_number * TBz[tt] * sqrt(2.0 * dU_B / intTB2);
            }

            //dh_check = 0.50*dx*arma::accu(Hy % Hy) - de_Hy;
            //cout << "Dh_CHECK_Hy = " << dh_check << endl;

        }
        else
        {
            double root1_B = (-intBTB + sqrt(intBTB*intBTB + 2.0 * dU_B * intTB2) )/ intTB2;
            double root2_B = (-intBTB - sqrt(intBTB*intBTB + 2.0 * dU_B * intTB2) )/ intTB2;
            //if (abs(root1_B) < abs(root2_B))
            //if (abs(root1_B + KAPPA_B)<abs(root2_B + KAPPA_B))
            //if (root1_B > 0.0)
            if (root1_B*kappa_E>0.0)
                kappa_B = root1_B;
            else
                kappa_B = root2_B;
            //dh_check = 0.50 * dx * kappa_Hy * kappa_Hy * arma::accu(dPzdx % dPzdx) - de_Hy;
            //cout << "Dh_CHECK_Hy = " << dh_check << " de_Hy = " << de_Hy << endl;
            //Hy += kappa_Hy * dPzdx;
            #pragma omp parallel for schedule(runtime)
            for(size_t tt=0; tt < N_GRIDS_3D; tt++)
            {
                x[Bx_START+tt] += kappa_B * TBx[tt];
                x[By_START+tt] += kappa_B * TBy[tt];
                x[Bz_START+tt] += kappa_B * TBz[tt];
            }
            KAPPA_B += kappa_B;

            //double energy_BR = 0.0;
            //#pragma omp parallel for reduction(+:energy_BR) schedule(runtime)
            //for(size_t tt=0; tt<N_GRIDS_3D; tt++)
            //{
            //energy_BR += x[tt+Bx_START]*x[tt+Bx_START] + x[tt+By_START]*x[tt+By_START] + x[tt+Bz_START]*x[tt+Bz_START];
            //}
            //energy_BR *= dR*dR*dR/2;
            //
            ////de_check = 0.50 * dx * kappa_Ez * kappa_Ez * arma::accu(Pz % Pz) - de_Ez;
            //cout << "DU_CHECK_B = " << energy_BR-energy_B << "; dU_B = " << dU_B << endl;

        }
    }

};



void integrate_runge_kutta4(ClassicalEM_2LevelSystem_3D &System, state_type &x, const int N, const double t_start, const double t_end, const double dt)
{
    state_type k1(N, 0.0), k2(N, 0.0), k3(N, 0.0), k4(N, 0.0);
    state_type x2(N, 0.0), x3(N, 0.0), x4(N, 0.0);
    for(double t=t_start; t<=t_end; t+=dt)
    {
        // Calculate k1
        System(x, k1, t);
        // Calculate k2
        #pragma omp parallel for schedule(runtime)
        for(size_t i=0; i<N; i++)
        {
            x2[i] = x[i] + dt/2.0*k1[i];
        }
        System(x2, k2, t+dt/2.0);
        // Calculate k3
        #pragma omp parallel for schedule(runtime)
        for(size_t i=0; i<N; i++)
        {
            x3[i] = x[i] + dt/2.0*k2[i];
        }
        System(x3, k3, t+dt/2.0);
        // Calculate k4
        #pragma omp parallel for schedule(runtime)
        for(size_t i=0; i<N; i++)
        {
            x4[i] = x[i] + dt*k3[i];
        }
        System(x4, k4, t+dt);
        // Update x
        #pragma omp parallel for schedule(runtime)
        for(size_t i=0; i<N; i++)
        {
            x[i] = x[i] + dt/6.0*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
        }
        
        System.augment(x, t);
    }
}
void EhrenfestDynamics(double c1_real, double c1_imag, double c2_real, double c2_imag, const string filename = "erhenfest_traj.txt")
{
    // Initialize for state_type
    state_type x(N_VARIABLES);
    x[0] = c1_real;
	x[1] = c1_imag;
	x[2] = c2_real;
	x[3] = c2_imag;
    for (int i = Ex_START; i != Bz_START + N_GRIDS_3D; i++)
		x[i] = 0.0;
    // Define ODE MODEL
    ClassicalEM_2LevelSystem_3D my_model(X_MAX, X_MIN, OMEGA, P_MAX, SIGMA);
    // Record the EM field distribution at different time
    //double start_time[5] = {0, 20, 40, 60, 80};
    //double end_time[5] = {20, 40, 60, 80, 100};
    //double start_time[5] = {0, 100*dt, 200*dt, 300*dt, 400*dt};
    //double end_time[5] = {100*dt, 200*dt, 300*dt, 400*dt, 500*dt};
    for(int count=0; count<5; count++)
    {
        // Integrate ODEs in a time range
        integrate_runge_kutta4(my_model, x, N_VARIABLES, start_time[count], end_time[count], dt);
        // output Electric field at the end_time
        ofstream myfile;
        myfile.open("field+EB_t_"+to_string(end_time[count])+".txt");
        for(int i=0; i<N_GRIDS_3D;i++)
        {
            myfile <<setprecision(7)<<scientific  << x[Ex_START+i]<<" "<<x[Ey_START+i]<<" "<<x[Ez_START+i]<<" "<< x[Bx_START+i]<<" "<<x[By_START+i]<<" "<<x[Bz_START+i] <<"\n";
        }
        myfile.close();
    }
}

int main()
{
    int chunk_size = N_GRIDS_3D/omp_get_max_threads();
    omp_set_schedule( omp_sched_static , chunk_size );
    EhrenfestDynamics(sqrt(C0), 0.0, sqrt(1.0-C0), 0.0, "traj_Np_"+to_string(N_GRIDS_1D)+".txt");
    return 0;
}


// generate a 0-1 distributed float number using mt19937 method
double get_random_mt19937()
{
	static std::mt19937 e;
	static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
	return dis(e);
}

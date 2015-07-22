#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include <stdio.h>

// Add interlayer exchange field to B.
extern "C" __global__ void
addinterlayerexchange(float* __restrict__  Bx, float* __restrict__  By, float* __restrict__  Bz,
                      float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
                      float* __restrict__ J1LUT, float* __restrict__ J2LUT,
                      float* __restrict__ toplayer,  float* __restrict__ bottomlayer,
                      float* __restrict__ uxLUT, float* __restrict__ uyLUT, float* __restrict__ uzLUT,
		      float cx, float cy, float cz, int Nx, int Ny, int Nz, uint8_t* __restrict__ regions) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;
	int i = idx(ix, iy, iz);
	if (ix >= Nx || iy >= Ny || iz >= Nz) {
		return;
	}
	
	uint8_t reg = regions[i];
	
    float ux = uxLUT[reg];
	float uy = uyLUT[reg];
	float uz = uzLUT[reg];
	float3 u = normalized(make_float3(ux, uy, uz));

	float J_iec1  = J1LUT[reg];
	float J_iec2  = J2LUT[reg];

	int top = __double2int_rn(toplayer[reg]);
	int bottom = __double2int_rn(bottomlayer[reg];	
	
	if (__double2int_rn(u.x) == 0 && __double2int_rn(u.y) == 0 && __double2int_rn(u.z) == 1) {
		//if (ix > Nx || iy > Ny || iz > Nz) return;
		//printf("%d \t %d \t %d \t %d \n", i, ix, iy ,iz);
		//
	
		int start_pos_bot = Nx*Ny*(bottom);
		int end_pos_bot = Nx*Ny*(bottom) + Nx*Ny-1;

		int start_pos_top = Nx*Ny*(top);
		int end_pos_top = Nx*Ny*(top) + Nx*Ny-1; 
		if ( i >= start_pos_top && i <= end_pos_top) {
			int i_below = i - (top-bottom)*Nx*Ny;
			float3 m   = {mx[i], my[i], mz[i]};
			float3 m_prime = { mx[i_below], my[i_below], mz[i_below]};
			if ( is0(m) || is0(m_prime)) {
				return;
			}
			float3 Biec_top  = (J_iec1 * m_prime + 2.0f * J_iec2 * m_prime * dot(m, m_prime)); // calc. IEC in toplayer
		
			Bx[i] += Biec_top.x/cz;
			By[i] += Biec_top.y/cz;
			Bz[i] += Biec_top.z/cz;	
			//printf("%e \n", cellsize_z);
			//printf("%e \t %e \t %e\n", Biec_top.x,Biec_top.y,Biec_top.z);
		}

		if ( i >= start_pos_bot && i <= end_pos_bot) {
			int i_above = i + (top-bottom)*Nx*Ny;
			float3 m   = {mx[i_above], my[i_above], mz[i_above]};
			float3 m_prime = { mx[i], my[i], mz[i]};		
			if ( is0(m) || is0(m_prime)) {
				return;
			}
			float3 Biec_bottom  = (J_iec1 * m + 2.0f * J_iec2 * m * dot(m_prime, m)); // calc. IEC in bottomlayer		
			//float3 Biec_bottom  = (J_iec1 * m + 2.0f * J_iec2 * m * dot(m, m_prime)); // calc. IEC in bottomlayer		

			Bx[i] += Biec_bottom.x/cz;
			By[i] += Biec_bottom.y/cz;
			Bz[i] += Biec_bottom.z/cz;
			//printf("%e \t %e \t %e\n", Bx[i],By[i],Bz[i]);
		}
	}
	if (__double2int_rn(u.x) == 0 && __double2int_rn(u.y) == 1 && __double2int_rn(u.z) == 0) {
		//int i_y =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
		//if (ix > Nx || iy > Ny || iz > Nz) return;
		//printf("%d \t %d \t %d \t %d \n", i, ix, iy ,iz);
		float  cellsize_y = cy;
	
		int start_pos_bot = (bottom+1)*Nx-Nx;
		int end_pos_bot = start_pos_bot + Nx-1;

		int start_pos_top = (top+1)*Nx-Nx;
		int end_pos_top = start_pos_top + Nx-1;
		//printf("%d \t %d \n", start_pos_bot, end_pos_bot);
		//printf("%d \t %d \n",start_pos_top, end_pos_top); 
		if ( i >= start_pos_top +iz*Nx*Ny && i <= end_pos_top +iz*Nx*Ny) {
		//if ( i >= start_pos_top && i <= end_pos_top) {
			int i_below = i - (top-bottom)*Nx;
			float3 m   = {mx[i], my[i], mz[i]};
			float3 m_prime = { mx[i_below], my[i_below], mz[i_below]};
			if ( is0(m) || is0(m_prime)) {
				return;
			}
			float3 Biec_top  = (J_iec1 * m_prime + 2.0f * J_iec2 * m_prime * dot(m, m_prime)); // calc. IEC in toplayer
			//printf("%e \n", cellsize_y);
			//printf("%e \t %e \t %e\n", Biec_top.x,Biec_top.y,Biec_top.z);
			Bx[i] += Biec_top.x/cy;
			By[i] += Biec_top.y/cy;
			Bz[i] += Biec_top.z/cy;
		}

		if ( i >= start_pos_bot+iz*Nx*Ny && i <= end_pos_bot+iz*Nx*Ny) {
		//if ( i >= start_pos_bot && i <= end_pos_bot)	{
			int i_above = i + (top-bottom)*Nx;
			//printf("%d \t %d \n", i, i_above);
			float3 m   = {mx[i_above], my[i_above], mz[i_above]};
			float3 m_prime = { mx[i], my[i], mz[i]};		
			if ( is0(m) || is0(m_prime)) {
				return;
			}
			float3 Biec_bottom  = (J_iec1 * m + 2.0f * J_iec2 * m * dot(m_prime, m)); // calc. IEC in bottomlayer		
			//float3 Biec_bottom  = (J_iec1 * m + 2.0f * J_iec2 * m * dot(m, m_prime)); // calc. IEC in bottomlayer		
			Bx[i] += Biec_bottom.x/cy;
			By[i] += Biec_bottom.y/cy;
			Bz[i] += Biec_bottom.z/cy;
			//printf("%e \t %e \t %e\n", Bx[i],By[i],Bz[i]);
		}
	}
	if (__double2int_rn(u.x) == 1 && __double2int_rn(u.y) == 0 && __double2int_rn(u.z) == 0) {
		float  cellsize_x = cx;
	
		if ( i == bottom + iy*Nx + iz*Nx*Ny   && i < Nx*Ny*Nz) {
			int i_above = i + (top-bottom);
			//printf("%d \t %d \n", i, i_above);
			float3 m_prime   = {mx[i], my[i], mz[i]};
			float3 m = { mx[i_above], my[i_above], mz[i_above]};
			if ( is0(m) || is0(m_prime)) {
				return;
			}
			float3 Biec_bottom  = (J_iec1 * m + 2.0f * J_iec2 * m * dot(m_prime, m)); // calc. IEC in bottomlayer
		
			Bx[i] += Biec_bottom.x/cx;
			By[i] += Biec_bottom.y/cx;
			Bz[i] += Biec_bottom.z/cx;
		}

		if ( i == top + iy*Nx + iz*Nx*Ny   && i < Nx*Ny*Nz) {
			int i_below = i - (top-bottom);
			float3 m   = {mx[i], my[i], mz[i]};
			float3 m_prime = { mx[i_below], my[i_below], mz[i_below]};		
			if ( is0(m) || is0(m_prime)) {
				return;
			}
			float3 Biec_top  = (J_iec1 * m_prime + 2.0f * J_iec2 * m_prime * dot(m, m_prime)); // calc. IEC in toplayer		
			//float3 Biec_bottom  = (J_iec1 * m + 2.0f * J_iec2 * m * dot(m, m_prime)); // calc. IEC in bottomlayer		
	
			Bx[i] += Biec_top.x/cx;
			By[i] += Biec_top.y/cx;
			Bz[i] += Biec_top.z/cx;
		}
	}
}


__global__ void KERNEL_set_alpha_sph(part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].i_type==3) return;

	uint_t ptype;
	ptype=P1[i].p_type;

	if(ptype<=1000){	// Eulerian SPH
		P1[i].elix=0.0;
		P1[i].eliy=0.0;
		P1[i].eliz=0.0;
	}
	else{	// DEM
		P1[i].elix=1.0;
		P1[i].eliy=1.0;
		P1[i].eliz=1.0;
	}


}

__global__ void KERNEL_set_alpha_dem(part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_dem) return;
	if(P1[i].i_type==3) return;

	uint_t ptype;
	ptype=P1[i].p_type;

	if(ptype<=1000){	// Eulerian SPH
		P1[i].elix=0.0;
		P1[i].eliy=0.0;
		P1[i].eliz=0.0;
	}
	else{	// DEM
		P1[i].elix=1.0;
		P1[i].eliy=1.0;
		P1[i].eliz=1.0;
	}
}






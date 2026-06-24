__global__ void KERNEL_clc_mass_init_sph(int_t*g_str,int_t*g_end,part1*P1,part2*P2,int_t tcount)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].p_type>1000) return;
	if(P1[i].i_type>i_type_crt) return;

		//P2[i].rho_ref0 = P1[i].rho;
		//P2[i].rho_ref = P1[i].rho;
		P1[i].vol0 = pow(P1[i].h/1.6,k_dim);
		P1[i].vol = P1[i].vol0;

		P2[i].rho0=P1[i].rho;

		Real tempi = P1[i].temp;
		Real ptypei = P1[i].p_type;
		P1[i].cond = conductivity(tempi,ptypei);


}
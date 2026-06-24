#define L2	-0.021
#define L3	0.006

__host__ __device__ uint64_t morton2d(uint64_t x,uint64_t y)
{
	uint64_t z=0;

	x=(x|(x<<16))&0x0000FFFF0000FFFF;
	x=(x|(x<<8))&0x00FF00FF00FF00FF;
	x=(x|(x<<4))&0x0F0F0F0F0F0F0F0F;
	x=(x|(x<<2))&0x3333333333333333;
	x=(x|(x<<1))&0x5555555555555555;
	y=(y|(y<<16))&0x0000FFFF0000FFFF;
	y=(y|(y<<8))&0x00FF00FF00FF00FF;
	y=(y|(y<<4))&0x0F0F0F0F0F0F0F0F;
	y=(y|(y<<2))&0x3333333333333333;
	y=(y|(y<<1))&0x5555555555555555;

	z=x|(y<<1);

	return z;
}

__host__ __device__ uint64_t morton3d(unsigned int a,unsigned int b,unsigned int c)
{
	uint64_t answer=0;

	uint64_t x=a&0x1fffff;// we only look at the first 21 bits
	x=(x|x<<32)&0x1f00000000ffff; // shift left 32 bits,OR with self,and 00011111000000000000000000000000000000001111111111111111
	x=(x|x<<16)&0x1f0000ff0000ff; // shift left 32 bits,OR with self,and 00011111000000000000000011111111000000000000000011111111
	x=(x|x<<8)&0x100f00f00f00f00f;// shift left 32 bits,OR with self,and 0001000000001111000000001111000000001111000000001111000000000000
	x=(x|x<<4)&0x10c30c30c30c30c3;// shift left 32 bits,OR with self,and 0001000011000011000011000011000011000011000011000011000100000000
	x=(x|x<<2)&0x1249249249249249;

	uint64_t y=b&0x1fffff;// we only look at the first 21 bits
	y=(y|y<<32)&0x1f00000000ffff; // shift left 32 bits,OR with self,and 00011111000000000000000000000000000000001111111111111111
	y=(y|y<<16)&0x1f0000ff0000ff; // shift left 32 bits,OR with self,and 00011111000000000000000011111111000000000000000011111111
	y=(y|y<<8)&0x100f00f00f00f00f;// shift left 32 bits,OR with self,and 0001000000001111000000001111000000001111000000001111000000000000
	y=(y|y<<4)&0x10c30c30c30c30c3;// shift left 32 bits,OR with self,and 0001000011000011000011000011000011000011000011000011000100000000
	y=(y|y<<2)&0x1249249249249249;

	uint64_t z=c&0x1fffff;// we only look at the first 21 bits
	z=(z|z<<32)&0x1f00000000ffff; // shift left 32 bits,OR with self,and 00011111000000000000000000000000000000001111111111111111
	z=(z|z<<16)&0x1f0000ff0000ff; // shift left 32 bits,OR with self,and 00011111000000000000000011111111000000000000000011111111
	z=(z|z<<8)&0x100f00f00f00f00f;// shift left 32 bits,OR with self,and 0001000000001111000000001111000000001111000000001111000000000000
	z=(z|z<<4)&0x10c30c30c30c30c3;// shift left 32 bits,OR with self,and 0001000011000011000011000011000011000011000011000011000100000000
	z=(z|z<<2)&0x1249249249249249;

	answer|=x|y<<1|z<<2;
	return answer;
}

int_t clc_num_cells() 
{

	int_t result;
	int_t NI_max;

	NI_max=max(max(NI,NJ),NK);

	//printf("NI_max = %d\n", NI_max);

	if (flag_z_index==0) {
		result=NI*NJ*NK+1;
	}
	else {
		if (dim==2) result=morton2d(NI_max-1,NI_max-1);
		if (dim==3) result=morton3d(NI_max-1,NI_max-1,NI_max-1);
	}

	return result;
}

void c_initial_inner_outer_particle_single2(part1*HP1,part1*DHP1,int_t tid)
{

	int_t i,c_count;
	Real xi0;
	Real maxb,minb;

	c_count=0;

	for(i=0;i<num_part;i++){
		HP1[i].i_type=1;
		HP1[i].buffer_type=0;

		if (open_boundary>0 && HP1[i].p_type<1000)			// (CAUTION)
		{
			if(HP1[i].buffer_type>=1){
				HP1[i].i_type=2;
			}
		}
		DHP1[c_count]=HP1[i];
		c_count++;
	}

}

void c_initial_inner_outer_particle_single_sph2(part1*HP1,part1*DHP1,int_t tid)
{

	int_t i,c_count;
	Real xi0;
	Real maxb,minb;

	c_count=0;

	for(i=0;i<num_part_sph;i++){
		HP1[i].i_type=1;
		HP1[i].buffer_type=0;

		if (open_boundary>0 && HP1[i].p_type<1000)			// (CAUTION)
		{
			if(HP1[i].buffer_type>=1){
				HP1[i].i_type=2;
			}
		}
		DHP1[c_count]=HP1[i];
		c_count++;
	}
}

void c_initial_inner_outer_particle_single_dem2(part1*HP1,part1*DHP1,int_t tid)
{

	int_t i,c_count;
	Real xi0;
	Real maxb,minb;

	c_count=0;

	for(i=0;i<num_part_dem;i++){
		HP1[i].i_type=1;
		HP1[i].buffer_type=0;

		if (open_boundary>0 && HP1[i].p_type<1000)			// (CAUTION)
		{
			if(HP1[i].buffer_type>=1){
				HP1[i].i_type=2;
			}
		}
		DHP1[c_count]=HP1[i];
		c_count++;
	}
}

__host__ __device__ int_t idx_cell(int_t Ix, int_t Iy, int_t Iz) 
{

	int_t result;

	if (k_flag_z_index==0) {
		result=(Ix)+k_NI*(Iy)+k_NI*k_NJ*(Iz);
	}
	else {
		if (k_dim==2) result=morton2d(Ix,Iy);
		if (k_dim==3) result=morton3d(Ix,Iy,Iz);
	}
	return result;
}

__global__ void KERNEL_index_particle_to_cell_sph(int_t*g_idx,int_t*p_idx,part1*P1)
{
	int_t idx=threadIdx.x+blockIdx.x*blockDim.x;
	if(idx>=k_num_part2_sph) return;
	// Not in Each GPUs Domain
	if(P1[idx].i_type==3){
		g_idx[idx]=k_num_cells;
		p_idx[idx]=idx;
		return;
	}
	else {
		int_t icell,jcell,kcell;
		// calculate I,J,K in cell
		if((k_x_max==k_x_min)){icell=0;}
		else{icell=min(floor((P1[idx].x-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
		if((k_y_max==k_y_min)){jcell=0;}
		else{jcell=min(floor((P1[idx].y-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
		if((k_z_max==k_z_min)){kcell=0;}
		else{kcell=min(floor((P1[idx].z-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
		// out-of-range handling
		if(icell<0) icell=0;
		if(jcell<0) jcell=0;
		if(kcell<0) kcell=0;
		// calculate cell index from I,J,K
		p_idx[idx]=idx;
		g_idx[idx]=idx_cell(icell,jcell,kcell);
	}

}

__global__ void KERNEL_index_particle_to_cell_dem(int_t*g_idx,int_t*p_idx,part1*P1)
{
	int_t idx=threadIdx.x+blockIdx.x*blockDim.x;
	if(idx>=k_num_part2_dem) return;
	// Not in Each GPUs Domain
	if(P1[idx].i_type==3){
		g_idx[idx]=k_num_cells;
		p_idx[idx]=idx;
		return;
	}
	else {
		int_t icell,jcell,kcell;
		// calculate I,J,K in cell
		if((k_x_max==k_x_min)){icell=0;}
		else{icell=min(floor((P1[idx].x-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
		if((k_y_max==k_y_min)){jcell=0;}
		else{jcell=min(floor((P1[idx].y-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
		if((k_z_max==k_z_min)){kcell=0;}
		else{kcell=min(floor((P1[idx].z-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
		// out-of-range handling
		if(icell<0) icell=0;
		if(jcell<0) jcell=0;
		if(kcell<0) kcell=0;
		// calculate cell index from I,J,K
		p_idx[idx]=idx;
		g_idx[idx]=idx_cell(icell,jcell,kcell);
	}

}

__global__ void KERNEL_reorder_sph(int_t*g_idx,int_t*p_idx,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part1*SP1,part2*SP2)
{
	extern __shared__ int sharedHash[];

	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	if(idx>=k_num_part2_sph) return;
	int hash;

	hash=g_idx[idx];

	sharedHash[threadIdx.x+1]=hash;
	if(idx>0&&threadIdx.x==0){
		/*save the end of the previous block g_idx*/
		sharedHash[0]=g_idx[idx-1];
	}
	__syncthreads();		// for sorting and reorder particle property
	if(idx==0||hash!=sharedHash[threadIdx.x]){
		//if(hash<k_num_cells) {
			g_str[hash]=idx;
			if(idx>0) g_end[sharedHash[threadIdx.x]]=idx;
		//}
	}
	// if((idx==k_num_part2-1)&&(hash<k_num_cells)) g_end[hash]=idx+1;
	if((idx==k_num_part2_sph-1)) g_end[hash]=idx+1;
	/*reorder data*/
	int sortedIndex=p_idx[idx];

	SP1[idx]=P1[sortedIndex];
	SP2[idx]=P2[sortedIndex];
}

__global__ void KERNEL_reorder_dem(int_t*g_idx,int_t*p_idx,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part1*SP1,part2*SP2)
{
	extern __shared__ int sharedHash[];

	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	if(idx>=k_num_part2_dem) return;
	int hash;

	hash=g_idx[idx];

	sharedHash[threadIdx.x+1]=hash;
	if(idx>0&&threadIdx.x==0){
		/*save the end of the previous block g_idx*/
		sharedHash[0]=g_idx[idx-1];
	}
	__syncthreads();		// for sorting and reorder particle property
	if(idx==0||hash!=sharedHash[threadIdx.x]){
		//if(hash<k_num_cells) {
			g_str[hash]=idx;
			if(idx>0) g_end[sharedHash[threadIdx.x]]=idx;
		//}
	}
	// if((idx==k_num_part2-1)&&(hash<k_num_cells)) g_end[hash]=idx+1;
	if((idx==k_num_part2_dem-1)) g_end[hash]=idx+1;
	/*reorder data*/
	int sortedIndex=p_idx[idx];

	SP1[idx]=P1[sortedIndex];
	SP2[idx]=P2[sortedIndex];
}
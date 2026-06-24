#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err__), #call); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

//-------------------------------------------------------------------------------------------------
// 모든 입자 계산을 수행하는 주함수
//-------------------------------------------------------------------------------------------------
void SOPHIA_single_ISPH(int_t*g_idx,int_t*p_idx,int_t*g_idx_in,int_t*p_idx_in,int_t*g_str,int_t*g_end,
	part1*dev_P1,part1*dev_SP1,part2*dev_P2,part2*dev_SP2,part3*dev_P3,
	int_t*p2p_af_in,int_t*p2p_idx_in,int_t*p2p_af,int_t*p2p_idx,
	void*dev_sort_storage,size_t*sort_storage_bytes,part1*file_P1,part2*file_P2,part3*file_P3,int tid,int*aps_num_part,int*aps,Real*Cd,Real*Cl
	,Real*P0,Real*P1,Real*P2,Real*P3,

	int_t*g_idx_sph,int_t*p_idx_sph,int_t*g_idx_sph_in,int_t*p_idx_sph_in,int_t*g_str_sph,int_t*g_end_sph,
	part1*dev_P1_sph,part1*dev_SP1_sph,part2*dev_P2_sph,part2*dev_SP2_sph,part3*dev_P3_sph,
	void*dev_sort_storage_sph,size_t*sort_storage_bytes_sph,

	int_t*g_idx_dem,int_t*p_idx_dem,int_t*g_idx_dem_in,int_t*p_idx_dem_in,int_t*g_str_dem,int_t*g_end_dem,
	part1*dev_P1_dem,part1*dev_SP1_dem,part2*dev_P2_dem,part2*dev_SP2_dem,part3*dev_P3_dem,
	void*dev_sort_storage_dem,size_t*sort_storage_bytes_dem,

	part1*file_P1_sph,part2*file_P2_sph,part3*file_P3_sph,
	part1*file_P1_dem,part2*file_P2_dem,part3*file_P3_dem)
{
// printf("Current loop count: %d\n",count);
dim3 b,t;
t.x=128;
b.x=(num_part2-1)/t.x+1;
int s=sizeof(int)*(t.x+1);



//-------------------------------------------------------------------------------------------------
// ESPH 준비
// Eulerian SPH setting
// SPH입자의 elix, eliy, eliz를 0으로, DEM입자의 elix, eliy, eliz를 1로 초기화
//-------------------------------------------------------------------------------------------------
// KERNEL_set_alpha<<<b,t>>>(dev_P1);
if((count%decouple_stride)==0){
	b.x=(num_part2_sph-1)/t.x+1;
	KERNEL_set_alpha_sph<<<b,t>>>(dev_P1_sph);
}
b.x=(num_part2_dem-1)/t.x+1;
KERNEL_set_alpha_dem<<<b,t>>>(dev_P1_dem);
cudaDeviceSynchronize();


//-------------------------------------------------------------------------------------------------
// 주변입자 검색
//-------------------------------------------------------------------------------------------------

// g_str을 리셋
// cudaMemset(g_str,cu_memset,sizeof(int_t)*num_cells);
if((count%decouple_stride)==0){
	cudaMemset(g_str_sph,cu_memset,sizeof(int_t)*num_cells);
}
cudaMemset(g_str_dem,cu_memset,sizeof(int_t)*num_cells);
cudaDeviceSynchronize();

// 입자의 셀번호 계산
// g_idx_in[idx]에는 idx에 해당하는 입자가 속한 cell번호가 저장
// p_idx_in[idx]에는 idx가 저장됨
// b.x=(num_part2-1)/t.x+1;
// KERNEL_index_particle_to_cell<<<b,t>>>(g_idx_in,p_idx_in,dev_P1);
if((count%decouple_stride)==0){
	b.x=(num_part2_sph-1)/t.x+1;
	KERNEL_index_particle_to_cell_sph<<<b,t>>>(g_idx_sph_in,p_idx_sph_in,dev_P1_sph);
}
b.x=(num_part2_dem-1)/t.x+1;
KERNEL_index_particle_to_cell_dem<<<b,t>>>(g_idx_dem_in,p_idx_dem_in,dev_P1_dem);
cudaDeviceSynchronize();

// 셀번호를 바탕으로 정렬
// g_idx_in과 p_idx_in을 g_idx와 p_idx에 정렬된 상태로 저장 (g_idx_in을 기준으로 오름차순)
// g_idx_in과 p_idx_in은 변하지 않음
// cub::DeviceRadixSort::SortPairs(dev_sort_storage,*sort_storage_bytes,g_idx_in,g_idx,p_idx_in,p_idx,num_part2);
if((count%decouple_stride)==0){
	cub::DeviceRadixSort::SortPairs(dev_sort_storage_sph,*sort_storage_bytes_sph,g_idx_sph_in,g_idx_sph,p_idx_sph_in,p_idx_sph,num_part2_sph);
}
cub::DeviceRadixSort::SortPairs(dev_sort_storage_dem,*sort_storage_bytes_dem,g_idx_dem_in,g_idx_dem,p_idx_dem_in,p_idx_dem,num_part2_dem);
cudaDeviceSynchronize();

// 정렬한 입자를 재배치
// dev_SP1, dev_SP2를 g_str, g_end를 기준으로 재배치
// b.x=(num_part2-1)/t.x+1;
// KERNEL_reorder<<<b,t,s>>>(g_idx,p_idx,g_str,g_end,dev_P1,dev_P2,dev_SP1,dev_SP2);
if((count%decouple_stride)==0){
	b.x=(num_part2_sph-1)/t.x+1;
	KERNEL_reorder_sph<<<b,t,s>>>(g_idx_sph,p_idx_sph,g_str_sph,g_end_sph,dev_P1_sph,dev_P2_sph,dev_SP1_sph,dev_SP2_sph);
}
b.x=(num_part2_dem-1)/t.x+1;
KERNEL_reorder_dem<<<b,t,s>>>(g_idx_dem,p_idx_dem,g_str_dem,g_end_dem,dev_P1_dem,dev_P2_dem,dev_SP1_dem,dev_SP2_dem);
cudaDeviceSynchronize();

// 일부 입자정보 리셋
// cudaMemset(dev_P3,0,sizeof(part3)*num_part2);
if((count%decouple_stride)==0){
	cudaMemset(dev_P3_sph,0,sizeof(part3)*num_part2_sph);
}
cudaMemset(dev_P3_dem,0,sizeof(part3)*num_part2_dem);

// 입자정보를 P1 에 복사
// Cell index기준으로 정렬된 입자정보를 dev_P1에 복사
// cudaMemcpy(dev_P1,dev_SP1,sizeof(part1)*num_part2,cudaMemcpyDeviceToDevice);
if((count%decouple_stride)==0){
	cudaMemcpy(dev_P1_sph,dev_SP1_sph,sizeof(part1)*num_part2_sph,cudaMemcpyDeviceToDevice);
}
cudaMemcpy(dev_P1_dem,dev_SP1_dem,sizeof(part1)*num_part2_dem,cudaMemcpyDeviceToDevice);
pthread_barrier_wait(&barrier);


//----------------------------------------------------------------
// 물리량 준비해놓기
// SPH입자에 대해서 SP1.vol, vol0에 (h/1.6)^3이 저장, SP1.cond에 conductivity가 저장
//----------------------------------------------------------------
// b.x=(num_part2-1)/t.x+1;
// if(dim==3) KERNEL_clc_mass_init<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,count);
if((count%decouple_stride)==0){
	b.x=(num_part2_sph-1)/t.x+1;
	if(dim==3) KERNEL_clc_mass_init_sph<<<b,t>>>(g_str_sph,g_end_sph,dev_SP1_sph,dev_SP2_sph,count);
}
cudaDeviceSynchronize();


//----------------------------------------------------------------
// 미분보정항 계산
// SPH입자에 대해서 dev_P3.CM에 3x3행렬값 저장
//----------------------------------------------------------------
// if(kgc_solve>0)	gradient_correction(g_str,g_end,dev_SP1,dev_P3);
if((count%decouple_stride)==0){
	if(kgc_solve>0){
		gradient_correction_sph(g_str_sph,g_end_sph,dev_SP1_sph,dev_P3_sph);
	}
}


//----------------------------------------------------------------
// SPH-DEM coupling을 위한 porosity와 filter값 계산
// SPH 입자에는 dev_SP1.DEMpor값(epsilon_i)이 계산되어 들어감
// DEM 입자에는 dev_SP1.filt_s값(이웃SPH의 sum(W*m/rho)), filt_sd(filt_s와 동일), filt_sd_2(SPH입자 종류가 여러개이면 1번종류만의 sum(W*m/rho))값이 계산되어 들어감
//----------------------------------------------------------------
// b.x=(num_part2-1)/t.x+1;
// if(dim==3) KERNEL_clc_prep3D_coupling<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2, count, time);
if((count%decouple_stride)==0){
	b.x=(num_part2_sph-1)/t.x+1;
	if(dim==3) KERNEL_clc_prep3D_coupling_sph<<<b,t>>>(g_str_sph,g_end_sph,dev_SP1_sph,dev_SP2_sph, g_str_dem,g_end_dem, dev_SP1_dem,dev_SP2_dem, count, time);
}
b.x=(num_part2_dem-1)/t.x+1;
if(dim==3) KERNEL_clc_prep3D_coupling_dem<<<b,t>>>(g_str_dem,g_end_dem,dev_SP1_dem,dev_SP2_dem, g_str_sph,g_end_sph, dev_SP1_sph,dev_SP2_sph, count, time);
cudaDeviceSynchronize();


//----------------------------------------------------------------
// Shepard filter값, delta SPH위한 값 계산
// SPH 입자에는 dev_SP1.filts값이 주기마다 계산되어 들어감
// delta SPH를 위해서 dev_SP1.grad_rhox, dev_SP1.grad_rhoy, dev_SP1.grad_rhoz 값이 계산되어 들어감
// filter, reference density, p_type switch, penetration, normal gradient etc
//----------------------------------------------------------------
if(dim==3) {
	// b.x=(num_part2-1)/t.x+1;
	// KERNEL_clc_prep3D_prep<<<b,t>>>(g_str,g_end,dev_SP1, dev_SP2, dev_P3, count);
	if((count%decouple_stride)==0){
		b.x=(num_part2_sph-1)/t.x+1;
		KERNEL_clc_prep3D_prep_sph<<<b,t>>>(g_str_sph,g_end_sph,dev_SP1_sph, dev_SP2_sph, dev_P3_sph, count);
	}
	cudaDeviceSynchronize();

	// b.x=(num_part2-1)/t.x+1;
	// KERNEL_clc_prep3D<<<b,t>>>(g_str,g_end,dev_SP1, dev_SP2, dev_P3, count);
	if((count%decouple_stride)==0){
		b.x=(num_part2_sph-1)/t.x+1;
		KERNEL_clc_prep3D_sph<<<b,t>>>(g_str_sph,g_end_sph,dev_SP1_sph, dev_SP2_sph, dev_P3_sph, count);
	}
    cudaDeviceSynchronize();
}




//-------------------------------------------------------------------------------------------------
// advection에 의한 힘을 계산
// 이거 뭔지 모르겠음. Eulerian SPH쓰는거같은데 그게 뭔지도 모르겠음
//-------------------------------------------------------------------------------------------------
// b.x=(num_part2-1)/t.x+1;
// if(dim==2) KERNEL_advection_force2D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
// if(dim==3) KERNEL_advection_force3D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
if((count%decouple_stride)==0){
	b.x=(num_part2_sph-1)/t.x+1;
	if(dim==3) KERNEL_advection_force3D_sph<<<b,t>>>(1,g_str_sph,g_end_sph,dev_SP1_sph,dev_SP2_sph,dev_P3_sph);
}
cudaDeviceSynchronize();



//-------------------------------------------------------------------------------------------------
// DEM입자끼리의 contact에 의한 힘을 계산
// DEM입자에 대해서 dev_P3.ftotalx y z, dev_P3.torqx y z에 DEM입자끼리의 contact에 의한 힘과 토크가 들어감
//-------------------------------------------------------------------------------------------------
// b.x=(num_part2-1)/t.x+1;
// if(dim==3) KERNEL_DEM_interaction3D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_P1,dev_P3,time);		// ** (DEM Dyanamics, heat transfer)
b.x=(num_part2_dem-1)/t.x+1;
if(dim==3) KERNEL_DEM_interaction3D_dem<<<b,t>>>(1,g_str_dem,g_end_dem,dev_SP1_dem,dev_P1_dem,dev_P3_dem,time);
cudaDeviceSynchronize();



//-------------------------------------------------------------------------------------------------
// [SPH-DEM Coupling] DEM 입자 힘 계산 (압력구배힘, 항력 등) [영역 0]
// dev_SP1[i].Fdx_da에 DEM입자에 작용하는 SPH에 의한 Drag force (F^D_a) 계산
// dev_SP1[i].Fdx_b에 DEM입자에 작용하는 SPH에 의한 Pressure-Gradient force (F^P_a) 계산
// 근데 내부의 세부적인 계산에 대해서는 뭔가 이해를 못하겠음
//-------------------------------------------------------------------------------------------------
// b.x=(num_part2-1)/t.x+1;
//if(dim==2) KERNEL_DEM_coupling2D<<<b,t>>>(0,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
// if(dim==3) KERNEL_DEM_coupling3D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
b.x=(num_part2_sph-1)/t.x+1;
if(dim==3) KERNEL_DEM_coupling3D_dem<<<b,t>>>(1,g_str_dem,g_end_dem,dev_SP1_dem,dev_SP2_dem,dev_P3_dem,
										g_str_sph,g_end_sph,dev_SP1_sph,dev_SP2_sph,dev_P3_sph);
cudaDeviceSynchronize();


// -------------------------------------------------------------------------------------------------
// [SPH-DEM Coupling] SPH 입자 힘 계산 (작용반작용) [영역 0]
// sph입자에 대해서 dev_SP1[i].ftotalx에 DEM입자에 의한 S_i항을 추가해서 더해줌
// -------------------------------------------------------------------------------------------------
// b.x=(num_part2-1)/t.x+1;
//if(dim==2) KERNEL_SPH_coupling2D<<<b,t>>>(0,g_str,g_end,dev_SP1,dev_P3);
// if(dim==3) KERNEL_SPH_coupling3D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_P3);
if((count%decouple_stride)==0){
	b.x=(num_part2_sph-1)/t.x+1;
	if(dim==3) KERNEL_SPH_coupling3D_sph<<<b,t>>>(1,g_str_sph,g_end_sph,dev_SP1_sph,dev_P3_sph,
											g_str_dem,g_end_dem,dev_SP1_dem,dev_P3_dem);
}
cudaDeviceSynchronize();

	
//-------------------------------------------------------------------------------------------------
// PREDICTOR (Optional)
// ISPH의 첫번째 velocity update를 진행해서 모든 입자의 dev_SP1.ux에 속도를 업데이트
// dev_SP1.x_star에 위 속도를 기반으로 한 위치 정보 저장
// dev_SP1.x에는 변하지 않은 원래의 위치가 그대로 기록되어있음
//-------------------------------------------------------------------------------------------------
if(time_type==Pre_Cor){
	// b.x=(num_part2-1)/t.x+1;
	// KERNEL_clc_projection<<<b,t>>>(count,dt,time,dev_SP1,dev_SP2,dev_P3);
	if((count%decouple_stride)==0){
		b.x=(num_part2_sph-1)/t.x+1;
		KERNEL_clc_projection_sph<<<b,t>>>(count,decouple_stride*dt,time,dev_SP1_sph,dev_SP2_sph,dev_P3_sph);
	}
	b.x=(num_part2_dem-1)/t.x+1;
	KERNEL_clc_projection_dem<<<b,t>>>(count,dt,time,dev_SP1_dem,dev_SP2_dem,dev_P3_dem);
	cudaDeviceSynchronize();
}



//-------------------------------------------------------------------------------------------------
// 압력 계산
// 벽면입자의 압력을 generalized pressure condition으로 설정
// Neumann boundary condition과는 아무런 관련 없음 이름만 이런거임
//-------------------------------------------------------------------------------------------------
// For initial condition
if(count==0){
	// b.x=(num_part2-1)/t.x+1;
	// if(dim==3)	KERNEL_Neumann_boundary3D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);
	if((count%decouple_stride)==0){
		b.x=(num_part2_sph-1)/t.x+1;
		if(dim==3)	KERNEL_Neumann_boundary3D_sph<<<b,t>>>(g_str_sph,g_end_sph,dev_SP1_sph,dev_SP2_sph,dev_P3_sph);
	}
    cudaDeviceSynchronize();
}



//-------------------------------------------------------------------------------------------------
// PPE solver
// PPE로 압력을 풀어 dev_SP1.pres에 압력을 계산해서 저장
//-------------------------------------------------------------------------------------------------
//if(dim==2)	KERNEL_PPE2D<<<b,t>>>(dt,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
// b.x=(num_part2-1)/t.x+1;
// if(dim==3)	KERNEL_PPE3D<<<b,t>>>(dt,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
if((count%decouple_stride)==0){
	b.x=(num_part2_sph-1)/t.x+1;
	if(dim==3)	KERNEL_PPE3D_sph<<<b,t>>>(decouple_stride*dt,g_str_sph,g_end_sph,dev_SP1_sph,dev_SP2_sph,dev_P3_sph);
}
cudaDeviceSynchronize();

// b.x=(num_part2-1)/t.x+1;
// if(dim==3)	KERNEL_PPE3D_calc<<<b,t>>>(dt,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
if((count%decouple_stride)==0){
	b.x=(num_part2_sph-1)/t.x+1;
	if(dim==3)	KERNEL_PPE3D_calc_sph<<<b,t>>>(decouple_stride*dt,g_str_sph,g_end_sph,dev_SP1_sph,dev_SP2_sph,dev_P3_sph);
}
cudaDeviceSynchronize();





//-------------------------------------------------------------------------------------------------
// 압력 계산
// 벽면입자의 압력을 generalized pressure condition으로 설정
// Neumann boundary condition과는 아무런 관련 없음 이름만 이런거임
//-------------------------------------------------------------------------------------------------
// Pressure condition for wall
// b.x=(num_part2-1)/t.x+1;
// if(dim==3)	KERNEL_Neumann_boundary3D<<<b,t>>>(g_str,g_end,dev_SP1,dev_SP2,dev_P3);
if(dim==3){
	if((count%decouple_stride)==0){
		b.x=(num_part2_sph-1)/t.x+1;
		if(dim==3)	KERNEL_Neumann_boundary3D_sph<<<b,t>>>(g_str_sph,g_end_sph,dev_SP1_sph,dev_SP2_sph,dev_P3_sph);
	}
}
cudaDeviceSynchronize();




//-------------------------------------------------------------------------------------------------
// 압력힘 계산
// 앞서 계산한 pres를 바탕으로 dev_P3.fpx에 pres force를 계산하여 저장함
//-------------------------------------------------------------------------------------------------
// b.x=(num_part2-1)/t.x+1;
// if(dim==2) KERNEL_pressureforce2D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
// if(dim==3) KERNEL_pressureforce3D<<<b,t>>>(1,g_str,g_end,dev_SP1,dev_SP2,dev_P3);
if((count%decouple_stride)==0){
	b.x=(num_part2_sph-1)/t.x+1;
	if(dim==3) KERNEL_pressureforce3D_sph<<<b,t>>>(1,g_str_sph,g_end_sph,dev_SP1_sph,dev_SP2_sph,dev_P3_sph);
}
cudaDeviceSynchronize();



//-------------------------------------------------------------------------------------------------
// 시간 적분 (Time Integration)
// 이때까지 계산한 값들을 바탕으로 dev_P1.x ux에 값을 업데이트해서 저장
//-------------------------------------------------------------------------------------------------
// b.x=(num_part2-1)/t.x+1;
// KERNEL_time_update_projection<<<b,t>>>(dt,dev_SP1,dev_P1,dev_SP2,dev_P2,dev_P3);
if((count%decouple_stride)==0){
	b.x=(num_part2_sph-1)/t.x+1;
	KERNEL_time_update_projection_sph<<<b,t>>>(decouple_stride*dt,dev_SP1_sph,dev_P1_sph,dev_SP2_sph,dev_P2_sph,dev_P3_sph);
}
b.x=(num_part2_dem-1)/t.x+1;
KERNEL_time_update_projection_dem<<<b,t>>>(dt,dev_SP1_dem,dev_P1_dem,dev_SP2_dem,dev_P2_dem,dev_P3_dem);
cudaDeviceSynchronize();

//-------------------------------------------------------------------------------------------------
// Open Boudnary
// Inlet입자와 Outlet입자의 물성치를 경계조건을 이용해서 바꿔줌
//-------------------------------------------------------------------------------------------------
if(open_boundary>0)
{
	// b.x=(num_part2-1)/t.x+1;
	// KERNEL_open_boundary_extrapolation3D_1<<<b,t>>>(time, g_str,g_end,dev_P1,dev_SP2,dev_P3,count,dt);
	if((count%decouple_stride)==0){
		b.x=(num_part2_sph-1)/t.x+1;
		KERNEL_open_boundary_extrapolation3D_1_sph<<<b,t>>>(time, g_str_sph,g_end_sph,dev_P1_sph,dev_SP2_sph,dev_P3_sph,count,decouple_stride*dt);
	}
    cudaDeviceSynchronize();

	// b.x=(num_part2-1)/t.x+1;
	// KERNEL_open_boundary_extrapolation3D_1_calc<<<b,t>>>(time, g_str,g_end,dev_P1,dev_SP2,dev_P3,count,dt);
	if((count%decouple_stride)==0){
		b.x=(num_part2_sph-1)/t.x+1;
		KERNEL_open_boundary_extrapolation3D_1_calc_sph<<<b,t>>>(time, g_str_sph,g_end_sph,dev_P1_sph,dev_SP2_sph,dev_P3_sph,count,decouple_stride*dt);
	}
    cudaDeviceSynchronize();
}




//-------------------------------------------------------------------------------------------------
// DEM 경계 조건 (DEM Boundary Condition) [영역 1]
// DEM입자의 벽면 충돌에 따라 Collision 또는 Rolling&Sliding을 계산해서 dev_P1.ux wx를 계산해서 저장
//-------------------------------------------------------------------------------------------------
b.x=(num_part2_dem-1)/t.x+1;
KERNEL_treat_DEM_box_x_dem<<<b,t>>>(1,dev_SP1_dem,dev_P1_dem,dev_SP2_dem,dev_P3_dem);
KERNEL_treat_DEM_box_y_dem<<<b,t>>>(1,dev_SP1_dem,dev_P1_dem,dev_SP2_dem,dev_P3_dem);
cudaDeviceSynchronize();



if(xsph_solve)
{
	// b.x=(num_part2-1)/t.x+1;
	// if(dim==2) KERNEL_xsph2D<<<b,t>>>(1,g_str,g_end,dt,time,dev_SP1,dev_P1,dev_SP2);
	// if(dim==3) KERNEL_xsph3D<<<b,t>>>(1,g_str,g_end,dt,time,dev_P1,dev_SP2);
	if((count%decouple_stride)==0){
		b.x=(num_part2_sph-1)/t.x+1;
		if(dim==3) KERNEL_xsph3D_sph<<<b,t>>>(1,g_str_sph,g_end_sph,decouple_stride*dt,time,dev_P1_sph,dev_SP2_sph);
	}
    cudaDeviceSynchronize();

	// b.x=(num_part2-1)/t.x+1;
	// if(dim==3) KERNEL_xsph3D_calc<<<b,t>>>(1,g_str,g_end,dt,time,dev_P1,dev_SP2);
	if((count%decouple_stride)==0){
		b.x=(num_part2_sph-1)/t.x+1;
		if(dim==3) KERNEL_xsph3D_calc_sph<<<b,t>>>(1,g_str_sph,g_end_sph,decouple_stride*dt,time,dev_P1_sph,dev_SP2_sph);
	}
    cudaDeviceSynchronize();
}
	//-------------------------------------------------------------------------------------------------
	// 출력
	//-------------------------------------------------------------------------------------------------

//if(((count%freq_output)==0) && (count>=0)){
if(((count%freq_output)==0) ){
        printf("save plot...........................\n");
        cudaMemcpy(file_P1,dev_P1,num_part2*sizeof(part1),cudaMemcpyDeviceToHost);
        cudaMemcpy(file_P2,dev_SP2,num_part2*sizeof(part2),cudaMemcpyDeviceToHost);
        cudaMemcpy(file_P3,dev_P3,num_part2*sizeof(part3),cudaMemcpyDeviceToHost);
        
        cudaMemcpy(file_P1_sph,dev_P1_sph,num_part2_sph*sizeof(part1),cudaMemcpyDeviceToHost);
        cudaMemcpy(file_P2_sph,dev_SP2_sph,num_part2_sph*sizeof(part2),cudaMemcpyDeviceToHost);
        cudaMemcpy(file_P3_sph,dev_P3_sph,num_part2_sph*sizeof(part3),cudaMemcpyDeviceToHost);

        cudaMemcpy(file_P1_dem,dev_P1_dem,num_part2_dem*sizeof(part1),cudaMemcpyDeviceToHost);
        cudaMemcpy(file_P2_dem,dev_SP2_dem,num_part2_dem*sizeof(part2),cudaMemcpyDeviceToHost);
        cudaMemcpy(file_P3_dem,dev_P3_dem,num_part2_dem*sizeof(part3),cudaMemcpyDeviceToHost);	

        //save_plot_fluid_vtk_bin_fluid(file_P1,file_P2,file_P3);		// fluid (SPH)
        //save_plot_fluid_vtk_bin_air(file_P1,file_P2,file_P3);
        // save_plot_fluid_vtk_bin_solid(file_P1,file_P3);		// solid (DEM)
		save_plot_fluid_vtk_bin_solid_dem(file_P1_dem,file_P3_dem);
    //	save_plot_fluid_vtk_bin_moving(file_P1,file_P3);
        //if (simulation_type==Two_Phase) save_plot_fluid_vtk_bin_fluid2(file_P1,file_P3);
        //if(count==0) save_plot_fluid_vtk_bin_boundary(file_P1);	// boundary (SPH) -> 맨 첫 스텝에만 출력
        //save_plot_fluid_vtk_bin_boundary(file_P1);	// boundary (SPH) -> 맨 첫 스텝에만 출력
        //save_plot_fluid_vtk_bin_boundary(file_P1);

        // save_vtk_bin_single(file_P1,file_P2,file_P3);
        // if(surf_model==2) CSF_validation(file_P1,file_P3);
        // if(surf_model==1) IPF_validation(file_P1,file_P3);
        printf("time = %5.6f\n\n\n",time);
}


//-------------------------------------------------------------------------------------------------
// 매 0.5초마다 현재 DEM 입자 상태를 input.txt 형식으로 저장 (input_generated 폴더)
//-------------------------------------------------------------------------------------------------
int save_input_stride = (int)(0.5/dt + 0.5);    // 0.5초에 해당하는 스텝 수
if(save_input_stride < 1) save_input_stride = 1;
if((count>0) && ((count%save_input_stride)==0)){
        cudaMemcpy(file_P1_dem,dev_P1_dem,num_part2_dem*sizeof(part1),cudaMemcpyDeviceToHost);
        save_input_dem(file_P1_dem);
}


}

//-------------------------------------------------------------------------------------------------
// SOPHIA 메인 코드
//-------------------------------------------------------------------------------------------------
void*ISPH_Calc(void*arg){

	// 함수의 인자를 받아서 tid 에 저장 (tid = gpu 번호)
	int*idPtr,tid;
	idPtr=(int*)arg;
	tid=*idPtr;

	// timestep control 을 위한 변수 설정
	Real dt_CFL,V_MAX,K_stiff,eta;
	Real h0=HP1[0].h;

	dt_CFL=V_MAX=K_stiff=eta=0.0;
	num_cells=clc_num_cells();

	count=floor(time/dt+0.5);


	//-------------------------------------------------------------------------------------------------
	// Device 입자 생성
	//-------------------------------------------------------------------------------------------------

	// 계산할 GPU 정의: tid=GPU number
	cudaSetDevice(tid);

	// GPU 내에 분기 생성 : stream is for run the kernel and memcpy peer to peer at the same time.
	cudaStreamCreate(&str1[tid]);
	cudaStreamCreate(&str2[tid]);

	// 출력할 변수 선언 및 메모리 할당
	part1*file_P1;
	file_P1=(part1*)malloc(sizeof(part1)*num_part2);
	memset(file_P1,0,sizeof(part1)*num_part2);
	part1*file_P1_sph;
	file_P1_sph=(part1*)malloc(sizeof(part1)*num_part2_sph);
	memset(file_P1_sph,0,sizeof(part1)*num_part2_sph);
	part1*file_P1_dem;
	file_P1_dem=(part1*)malloc(sizeof(part1)*num_part2_dem);
	memset(file_P1_dem,0,sizeof(part1)*num_part2_dem);

	part2*file_P2;
	part3*file_P3;
	part2*file_P2_sph;
	part3*file_P3_sph;
	part2*file_P2_dem;
	part3*file_P3_dem;

	file_P2=(part2*)malloc(sizeof(part2)*num_part2);
	memset(file_P2,0,sizeof(part2)*num_part2);
	file_P2_sph=(part2*)malloc(sizeof(part2)*num_part2_sph);
	memset(file_P2_sph,0,sizeof(part2)*num_part2_sph);
	file_P2_dem=(part2*)malloc(sizeof(part2)*num_part2_dem);
	memset(file_P2_dem,0,sizeof(part2)*num_part2_dem);

	
	file_P3=(part3*)malloc(sizeof(part3)*num_part2);
	memset(file_P3,0,sizeof(part3)*num_part2);
	file_P3_sph=(part3*)malloc(sizeof(part3)*num_part2_sph);
	memset(file_P3_sph,0,sizeof(part3)*num_part2_sph);
	file_P3_dem=(part3*)malloc(sizeof(part3)*num_part2_dem);
	memset(file_P3_dem,0,sizeof(part3)*num_part2_dem);

	//-------------------------------------------------------------------------------------------------
	// Device/GPU 변수 선언 및 메모리 할당
	//-------------------------------------------------------------------------------------------------

	// NNPS 관련 변수
	int_t*g_idx,*p_idx,*g_idx_in,*p_idx_in,*g_str,*g_end;
	int_t*g_idx_sph,*p_idx_sph,*g_idx_sph_in,*p_idx_sph_in,*g_str_sph,*g_end_sph;
	int_t*g_idx_dem,*p_idx_dem,*g_idx_dem_in,*p_idx_dem_in,*g_str_dem,*g_end_dem;

	// 주요 입자 변수
	part1*dev_P1,*dev_SP1;
	part1*dev_P1_sph,*dev_SP1_sph;
	part1*dev_P1_dem,*dev_SP1_dem;

	part2*dev_P2,*dev_SP2;
	part2*dev_P2_sph,*dev_SP2_sph;
	part2*dev_P2_dem,*dev_SP2_dem;
	
	part3*dev_SP3;
	part3*dev_SP3_sph;
	part3*dev_SP3_dem;

	// P2P 데이터 변수 선언
	int*p2p_af_in,*p2p_idx_in,*p2p_af,*p2p_idx;

	// for Adaptive Particle Definement
	int*aps_num_part;
	cudaMalloc((void**)&aps_num_part,sizeof(int));
	cudaMemset(aps_num_part,k_num_part,sizeof(int));

	int*aps;
	cudaMalloc((void**)&aps,sizeof(int));
	cudaMemset(aps,0,sizeof(int));

	// NNPS 입자 메모리 할당
	CUDA_CHECK(cudaMalloc((void**)&g_idx,sizeof(int_t)*num_part2));		// 입자의 셀번호				[N]
	CUDA_CHECK(cudaMalloc((void**)&p_idx,sizeof(int_t)*num_part2));		// 입자의 인덱스				[N]
	CUDA_CHECK(cudaMalloc((void**)&g_idx_in,sizeof(int_t)*num_part2));	// 입자의 셀번호 (정렬 전)		 [N]
	CUDA_CHECK(cudaMalloc((void**)&p_idx_in,sizeof(int_t)*num_part2));	// 입자의 인덱스 (정렬 전)		 [N]
	CUDA_CHECK(cudaMalloc((void**)&g_str,sizeof(int_t)*num_cells));		// 각 셀의 첫번째 입자의 인덱스   [num_cells]
	CUDA_CHECK(cudaMalloc((void**)&g_end,sizeof(int_t)*num_cells));		// 각 셀의 마지막 입자의 인덱스   [num_cells]

	CUDA_CHECK(cudaMalloc((void**)&g_idx_sph,sizeof(int_t)*num_part2_sph));		// SPH 입자의 셀번호				[N]
	CUDA_CHECK(cudaMalloc((void**)&p_idx_sph,sizeof(int_t)*num_part2_sph));		// SPH 입자의 인덱스				[N]
	CUDA_CHECK(cudaMalloc((void**)&g_idx_sph_in,sizeof(int_t)*num_part2_sph));	// SPH 입자의 셀번호 (정렬 전)	 [N]
	CUDA_CHECK(cudaMalloc((void**)&p_idx_sph_in,sizeof(int_t)*num_part2_sph));	// SPH 입자의 인덱스 (정렬 전)	 [N]
	CUDA_CHECK(cudaMalloc((void**)&g_str_sph,sizeof(int_t)*num_cells));			// SPH 각 셀의 첫번째 입자의 인덱스   [num_cells]
	CUDA_CHECK(cudaMalloc((void**)&g_end_sph,sizeof(int_t)*num_cells));			// SPH 각 셀의 마지막 입자의 인덱스   [num_cells]

	CUDA_CHECK(cudaMalloc((void**)&g_idx_dem,sizeof(int_t)*num_part2_dem));		// DEM 입자의 셀번호				[N]
	CUDA_CHECK(cudaMalloc((void**)&p_idx_dem,sizeof(int_t)*num_part2_dem));		// DEM 입자의 인덱스				[N]
	CUDA_CHECK(cudaMalloc((void**)&g_idx_dem_in,sizeof(int_t)*num_part2_dem));	// DEM 입자의 셀번호 (정렬 전)	 [N]
	CUDA_CHECK(cudaMalloc((void**)&p_idx_dem_in,sizeof(int_t)*num_part2_dem));	// DEM 입자의 인덱스 (정렬 전)	 [N]
	CUDA_CHECK(cudaMalloc((void**)&g_str_dem,sizeof(int_t)*num_cells));			// DEM 각 셀의 첫번째 입자의 인덱스   [num_cells]
	CUDA_CHECK(cudaMalloc((void**)&g_end_dem,sizeof(int_t)*num_cells));			// DEM 각 셀의 마지막 입자의 인덱스   [num_cells]


	// Device 입자 데이터 메모리 할당
	CUDA_CHECK(cudaMalloc((void**)&dev_P1,sizeof(part1)*num_part2));
	CUDA_CHECK(cudaMalloc((void**)&dev_SP1,sizeof(part1)*num_part2));
	CUDA_CHECK(cudaMalloc((void**)&dev_P2,sizeof(part2)*num_part2));
	CUDA_CHECK(cudaMalloc((void**)&dev_SP2,sizeof(part2)*num_part2));
	CUDA_CHECK(cudaMalloc((void**)&dev_SP3,sizeof(part3)*num_part2));

	CUDA_CHECK(cudaMalloc((void**)&dev_P1_sph,sizeof(part1)*num_part2_sph));
	CUDA_CHECK(cudaMalloc((void**)&dev_SP1_sph,sizeof(part1)*num_part2_sph));
	CUDA_CHECK(cudaMalloc((void**)&dev_P2_sph,sizeof(part2)*num_part2_sph));
	CUDA_CHECK(cudaMalloc((void**)&dev_SP2_sph,sizeof(part2)*num_part2_sph));
	CUDA_CHECK(cudaMalloc((void**)&dev_SP3_sph,sizeof(part3)*num_part2_sph));

	CUDA_CHECK(cudaMalloc((void**)&dev_P1_dem,sizeof(part1)*num_part2_dem));
	CUDA_CHECK(cudaMalloc((void**)&dev_SP1_dem,sizeof(part1)*num_part2_dem));
	CUDA_CHECK(cudaMalloc((void**)&dev_P2_dem,sizeof(part2)*num_part2_dem));
	CUDA_CHECK(cudaMalloc((void**)&dev_SP2_dem,sizeof(part2)*num_part2_dem));
	CUDA_CHECK(cudaMalloc((void**)&dev_SP3_dem,sizeof(part3)*num_part2_dem));


	// NNPS 메모리 초기화
	CUDA_CHECK(cudaMemset(g_idx_in,0,sizeof(int_t)*num_part2));
	CUDA_CHECK(cudaMemset(p_idx_in,0,sizeof(int_t)*num_part2));
	CUDA_CHECK(cudaMemset(g_idx,0,sizeof(int_t)*num_part2));
	CUDA_CHECK(cudaMemset(p_idx,0,sizeof(int_t)*num_part2));
	CUDA_CHECK(cudaMemset(g_str,cu_memset,sizeof(int_t)*num_cells));
	CUDA_CHECK(cudaMemset(g_end,0,sizeof(int_t)*num_cells));

	CUDA_CHECK(cudaMemset(g_idx_sph_in,0,sizeof(int_t)*num_part2_sph));
	CUDA_CHECK(cudaMemset(p_idx_sph_in,0,sizeof(int_t)*num_part2_sph));
	CUDA_CHECK(cudaMemset(g_idx_sph,0,sizeof(int_t)*num_part2_sph));
	CUDA_CHECK(cudaMemset(p_idx_sph,0,sizeof(int_t)*num_part2_sph));
	CUDA_CHECK(cudaMemset(g_str_sph,cu_memset,sizeof(int_t)*num_cells));
	CUDA_CHECK(cudaMemset(g_end_sph,0,sizeof(int_t)*num_cells));

	CUDA_CHECK(cudaMemset(g_idx_dem_in,0,sizeof(int_t)*num_part2_dem));
	CUDA_CHECK(cudaMemset(p_idx_dem_in,0,sizeof(int_t)*num_part2_dem));
	CUDA_CHECK(cudaMemset(g_idx_dem,0,sizeof(int_t)*num_part2_dem));
	CUDA_CHECK(cudaMemset(p_idx_dem,0,sizeof(int_t)*num_part2_dem));
	CUDA_CHECK(cudaMemset(g_str_dem,cu_memset,sizeof(int_t)*num_cells));
	CUDA_CHECK(cudaMemset(g_end_dem,0,sizeof(int_t)*num_cells));
	

	// Device 입자 메모리 초기화
	CUDA_CHECK(cudaMemset(dev_P1,0,sizeof(part1)*num_part2));
	CUDA_CHECK(cudaMemset(dev_SP1,0,sizeof(part1)*num_part2));
	CUDA_CHECK(cudaMemset(dev_P2,0,sizeof(part2)*num_part2));
	CUDA_CHECK(cudaMemset(dev_SP2,0,sizeof(part2)*num_part2));
	CUDA_CHECK(cudaMemset(dev_SP3,0,sizeof(part3)*num_part2));

	CUDA_CHECK(cudaMemset(dev_P1_sph,0,sizeof(part1)*num_part2_sph));
	CUDA_CHECK(cudaMemset(dev_SP1_sph,0,sizeof(part1)*num_part2_sph));
	CUDA_CHECK(cudaMemset(dev_P2_sph,0,sizeof(part2)*num_part2_sph));
	CUDA_CHECK(cudaMemset(dev_SP2_sph,0,sizeof(part2)*num_part2_sph));
	CUDA_CHECK(cudaMemset(dev_SP3_sph,0,sizeof(part3)*num_part2_sph));

	CUDA_CHECK(cudaMemset(dev_P1_dem,0,sizeof(part1)*num_part2_dem));
	CUDA_CHECK(cudaMemset(dev_SP1_dem,0,sizeof(part1)*num_part2_dem));
	CUDA_CHECK(cudaMemset(dev_P2_dem,0,sizeof(part2)*num_part2_dem));
	CUDA_CHECK(cudaMemset(dev_SP2_dem,0,sizeof(part2)*num_part2_dem));
	CUDA_CHECK(cudaMemset(dev_SP3_dem,0,sizeof(part3)*num_part2_dem));

	//-------------------------------------------------------------------------------------------------
	// Device/GPU로 데이터 복사
	//-------------------------------------------------------------------------------------------------

	// Sovler 전역변수 Device로 복사
	cudaMemcpyToSymbol(k_vii,vii,sizeof(int_t)*vii_size);
	cudaMemcpyToSymbol(k_vif,vif,sizeof(Real)*vif_size);

	// 물성 Table 데이터 Device로 복사
	cudaMemcpyToSymbol(k_Tab_T,host_Tab_T,sizeof(Real)*table_size);
	cudaMemcpyToSymbol(k_Tab_h,host_Tab_h,sizeof(Real)*table_size);
	cudaMemcpyToSymbol(k_Tab_k,host_Tab_k,sizeof(Real)*table_size);
	cudaMemcpyToSymbol(k_Tab_vis,host_Tab_vis,sizeof(Real)*table_size);

	cudaMemcpyToSymbol(k_table_index,host_table_index,sizeof(int)*10);
	cudaMemcpyToSymbol(k_table_size,host_table_size,sizeof(int)*10);

	// Host 입자정보(HP1)를 분할하여(DHP1) Device로 복사(dev_P1)
	DHP1[tid]=(part1*)malloc(num_part2*sizeof(part1));
	DHP1_sph[tid]=(part1*)malloc(num_part2_sph*sizeof(part1));
	DHP1_dem[tid]=(part1*)malloc(num_part2_dem*sizeof(part1));
	memset(DHP1[tid],0,sizeof(part1)*num_part2);
	memset(DHP1_sph[tid],0,sizeof(part1)*num_part2_sph);
	memset(DHP1_dem[tid],0,sizeof(part1)*num_part2_dem);
	printf("part2: %d, part2_sph: %d, part2_dem: %d\n", num_part2, num_part2_sph, num_part2_dem);


	// 모든 입자를 더미로 초기화
	for(int i=0;i<num_part2;i++) DHP1[tid][i].i_type=3;
	for(int i=0;i<num_part2_sph;i++) DHP1_sph[tid][i].i_type=3;
	for(int i=0;i<num_part2_dem;i++) DHP1_dem[tid][i].i_type=3;

	c_initial_inner_outer_particle_single2(HP1,DHP1[tid],tid);											// (CAUTION)
	c_initial_inner_outer_particle_single_sph2(HP1_sph,DHP1_sph[tid],tid);			// (CAUTION)
	c_initial_inner_outer_particle_single_dem2(HP1_dem,DHP1_dem[tid],tid);			// (CAUTION)
	// HP1에 있는 데이터를 DHP1으로 복사
	// inlet입자와 outlet입자에 대해서 buffer type을 각각 1, 2로 설정하고, itype을 2로 설정
	CUDA_CHECK(cudaMemcpy(dev_P1,DHP1[tid],num_part2*sizeof(part1),cudaMemcpyHostToDevice));	// single gpu 이면 그냥 HP1을 device에 복사
	CUDA_CHECK(cudaMemcpy(dev_P1_sph,DHP1_sph[tid],num_part2_sph*sizeof(part1),cudaMemcpyHostToDevice));	// single gpu 이면 그냥 HP1_sph을 device에 복사
	CUDA_CHECK(cudaMemcpy(dev_P1_dem,DHP1_dem[tid],num_part2_dem*sizeof(part1),cudaMemcpyHostToDevice));	// single gpu 이면 그냥 HP1_dem을 device에 복사
	pthread_barrier_wait(&barrier);

	if(tid==0){
		printf("\n-----------------------------------------------------------\n");
		printf("GPU Domain Division Success\n");
		printf("-----------------------------------------------------------\n\n");
	}


	//-------------------------------------------------------------------------------------------------
	// 화면 출력용 기타 변수들 정의 및 메모리 할당 (최대속도, 최대힘 등)
	//-------------------------------------------------------------------------------------------------

	// host
	Real *max_umag0,*max_rho0,*max_ftotal0;
	Real *max_umag0_sph,*max_rho0_sph,*max_ftotal0_sph;
	Real *max_umag0_dem,*max_rho0_dem,*max_ftotal0_dem;

	max_umag0=(Real*)malloc(sizeof(Real));
	max_rho0=(Real*)malloc(sizeof(Real));
	max_ftotal0=(Real*)malloc(sizeof(Real));
	max_umag0[0]=max_ftotal0[0]=max_rho0[0]=0.0;

	max_umag0_sph=(Real*)malloc(sizeof(Real));
	max_rho0_sph=(Real*)malloc(sizeof(Real));
	max_ftotal0_sph=(Real*)malloc(sizeof(Real));
	max_umag0_sph[0]=max_ftotal0_sph[0]=max_rho0_sph[0]=0.0;

	max_umag0_dem=(Real*)malloc(sizeof(Real));
	max_rho0_dem=(Real*)malloc(sizeof(Real));
	max_ftotal0_dem=(Real*)malloc(sizeof(Real));
	max_umag0_dem[0]=max_ftotal0_dem[0]=max_rho0_dem[0]=0.0;

	// device
	Real*max_rho,*max_umag,*max_ft,*d_max_umag0,*d_max_rho0,*d_max_ftotal0;
	Real*max_rho_sph,*max_umag_sph,*max_ft_sph,*d_max_umag0_sph,*d_max_rho0_sph,*d_max_ftotal0_sph;
	Real*max_rho_dem,*max_umag_dem,*max_ft_dem,*d_max_umag0_dem,*d_max_rho0_dem,*d_max_ftotal0_dem;
	
	CUDA_CHECK(cudaMalloc((void**)&max_rho,sizeof(Real)*num_part2));
	CUDA_CHECK(cudaMalloc((void**)&max_umag,sizeof(Real)*num_part2));
	CUDA_CHECK(cudaMalloc((void**)&max_ft,sizeof(Real)*num_part2));
	CUDA_CHECK(cudaMalloc((void**)&d_max_umag0,sizeof(Real)));
	CUDA_CHECK(cudaMalloc((void**)&d_max_rho0,sizeof(Real)));
	CUDA_CHECK(cudaMalloc((void**)&d_max_ftotal0,sizeof(Real)));
	CUDA_CHECK(cudaMemset(max_umag,0,sizeof(Real)*num_part2));
	CUDA_CHECK(cudaMemset(max_rho,0,sizeof(Real)*num_part2));
	CUDA_CHECK(cudaMemset(max_ft,0,sizeof(Real)*num_part2));
	CUDA_CHECK(cudaMemset(d_max_umag0,0,sizeof(Real)));
	CUDA_CHECK(cudaMemset(d_max_rho0,0,sizeof(Real)));
	CUDA_CHECK(cudaMemset(d_max_ftotal0,0,sizeof(Real)));

	CUDA_CHECK(cudaMalloc((void**)&max_rho_sph,sizeof(Real)*num_part2_sph));
	CUDA_CHECK(cudaMalloc((void**)&max_umag_sph,sizeof(Real)*num_part2_sph));
	CUDA_CHECK(cudaMalloc((void**)&max_ft_sph,sizeof(Real)*num_part2_sph));
	CUDA_CHECK(cudaMalloc((void**)&d_max_umag0_sph,sizeof(Real)));
	CUDA_CHECK(cudaMalloc((void**)&d_max_rho0_sph,sizeof(Real)));
	CUDA_CHECK(cudaMalloc((void**)&d_max_ftotal0_sph,sizeof(Real)));
	CUDA_CHECK(cudaMemset(max_umag_sph,0,sizeof(Real)*num_part2_sph));
	CUDA_CHECK(cudaMemset(max_rho_sph,0,sizeof(Real)*num_part2_sph));
	CUDA_CHECK(cudaMemset(max_ft_sph,0,sizeof(Real)*num_part2_sph));
	CUDA_CHECK(cudaMemset(d_max_umag0_sph,0,sizeof(Real)));
	CUDA_CHECK(cudaMemset(d_max_rho0_sph,0,sizeof(Real)));
	CUDA_CHECK(cudaMemset(d_max_ftotal0_sph,0,sizeof(Real)));

	CUDA_CHECK(cudaMalloc((void**)&max_rho_dem,sizeof(Real)*num_part2_dem));
	CUDA_CHECK(cudaMalloc((void**)&max_umag_dem,sizeof(Real)*num_part2_dem));
	CUDA_CHECK(cudaMalloc((void**)&max_ft_dem,sizeof(Real)*num_part2_dem));
	CUDA_CHECK(cudaMalloc((void**)&d_max_umag0_dem,sizeof(Real)));
	CUDA_CHECK(cudaMalloc((void**)&d_max_rho0_dem,sizeof(Real)));
	CUDA_CHECK(cudaMalloc((void**)&d_max_ftotal0_dem,sizeof(Real)));
	CUDA_CHECK(cudaMemset(max_umag_dem,0,sizeof(Real)*num_part2_dem));
	CUDA_CHECK(cudaMemset(max_rho_dem,0,sizeof(Real)*num_part2_dem));
	CUDA_CHECK(cudaMemset(max_ft_dem,0,sizeof(Real)*num_part2_dem));
	CUDA_CHECK(cudaMemset(d_max_umag0_dem,0,sizeof(Real)));
	CUDA_CHECK(cudaMemset(d_max_rho0_dem,0,sizeof(Real)));
	CUDA_CHECK(cudaMemset(d_max_ftotal0_dem,0,sizeof(Real)));


	//-------------------------------------------------------------------------------------------------
	// 정렬(Sorting)을 위한 CUB 라이브러리 변수 준비
	//-------------------------------------------------------------------------------------------------

	// Sorting & Max variable to use CUB Library
	void*dev_sort_storage=NULL;
	void*dev_max_storage=NULL;
	void*dev_sort_storage_sph=NULL;
	void*dev_max_storage_sph=NULL;
	void*dev_sort_storage_dem=NULL;
	void*dev_max_storage_dem=NULL;
	size_t sort_storage_bytes=0;
	size_t max_storage_bytes=0;
	size_t sort_storage_bytes_sph=0;
	size_t max_storage_bytes_sph=0;
	size_t sort_storage_bytes_dem=0;
	size_t max_storage_bytes_dem=0;

	// Determine Sorting & Maximum Value Setting for Total Particle Data
	// SortPairs는 필요한 임시 버퍼의 크기를 계산하기 위해서 한번 돌려줘야함
	cub::DeviceRadixSort::SortPairs(dev_sort_storage,sort_storage_bytes,g_idx_in,g_idx,p_idx_in,p_idx,num_part2);
	cub::DeviceRadixSort::SortPairs(dev_sort_storage_sph,sort_storage_bytes_sph,g_idx_sph_in,g_idx_sph,p_idx_sph_in,p_idx_sph,num_part2_sph);
	cub::DeviceRadixSort::SortPairs(dev_sort_storage_dem,sort_storage_bytes_dem,g_idx_dem_in,g_idx_dem,p_idx_dem_in,p_idx_dem,num_part2_dem);
	cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_umag,d_max_umag0,num_part2);
	cub::DeviceReduce::Max(dev_max_storage_sph,max_storage_bytes_sph,max_umag_sph,d_max_umag0_sph,num_part2_sph);
	cub::DeviceReduce::Max(dev_max_storage_dem,max_storage_bytes_dem,max_umag_dem,d_max_umag0_dem,num_part2_dem);

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMalloc((void**)&dev_sort_storage,sort_storage_bytes));
	CUDA_CHECK(cudaMalloc((void**)&dev_max_storage,max_storage_bytes));
	CUDA_CHECK(cudaMalloc((void**)&dev_sort_storage_sph,sort_storage_bytes_sph));
	CUDA_CHECK(cudaMalloc((void**)&dev_max_storage_sph,max_storage_bytes_sph));
	CUDA_CHECK(cudaMalloc((void**)&dev_sort_storage_dem,sort_storage_bytes_dem));
	CUDA_CHECK(cudaMalloc((void**)&dev_max_storage_dem,max_storage_bytes_dem));
	pthread_barrier_wait(&barrier);

	//-------------------------------------------------------------------------------------------------
	// 코드 메인
	//-------------------------------------------------------------------------------------------------

	// 초기상태 및 설정 출력
	if(tid==0){
		printf("-----------------------------------------------------------\n");
		printf("Input Summary: \n");
		printf("-----------------------------------------------------------\n");
		printf("	Total number of particles=%d\n",num_part);
		printf("	Device number of particles=%d\n",num_part2);
		printf("	P2P number of particles=%d\n",num_p2p);
		printf("	NI=%d,	NJ=%d,	NK=%d\n",NI,NJ,NK);
		printf("-----------------------------------------------------------\n\n");
		// Input Check
		printf("-----------------------------------------------------------\n");
		printf("Input Check: \n");
		printf("-----------------------------------------------------------\n");
		// check Domain Status
		printf("x min, max : %f %f\n",x_min,x_max);
		printf("y min, max : %f %f\n",y_min,y_max);
		printf("z min, max : %f %f\n",z_min,z_max);
		printf("Cell Size(dcell) %f\n",dcell);
		printf("Number of Cells Per a GPU in x-direction(calc_area) %d\n",calc_area);
        printf("Decouple Stride %d\n",decouple_stride);
		printf("-----------------------------------------------------------\n\n");
		// print out status
		printf("\n");
		printf("-----------------------------\n");
		printf("Start Simultion!!\n");
		printf("-----------------------------\n");
		printf("\n");
		printf("decouple_stride = %d\n", decouple_stride);
	}
	pthread_barrier_wait(&barrier);

	//-------------------------------------------------------------------------------------------------
	// 코드 메인
	//-------------------------------------------------------------------------------------------------

	int_t N = time_end/dt/freq_output;
Real Cdp[N], Cdv[N], Cd[N];
Real Clp[N], Clv[N], Cl[N];
Real P0[N], P1[N], P2[N], P3[N];
for(int i=0; i<N; i++)
{
Cdp[i]=0;
Cdv[i]=0;
Cd[i]=0;
Clp[i]=0;
Clv[i]=0;
Cl[i]=0;
P0[i]=0.0;
P1[i]=0.0;
P2[i]=0.0;
P3[i]=0.0;
}

	// t.x=128;
	// b.x=(num_part2-1)/t.x+1;
	// KERNEL_clc_TemptoEnthalpy<<<b,t>>>(dev_P1,dev_P2);
	// cudaDeviceSynchronize();

	while(1){

		if(ngpu==1){
		SOPHIA_single_ISPH(g_idx,p_idx,g_idx_in,p_idx_in,g_str,g_end,dev_P1,dev_SP1,dev_P2,dev_SP2,dev_SP3,
					p2p_af_in,p2p_idx_in,p2p_af,p2p_idx,dev_sort_storage,&sort_storage_bytes,file_P1,file_P2,file_P3,tid,aps_num_part,aps,Cd,Cl
					,P0,P1,P2,P3,
					g_idx_sph,p_idx_sph,g_idx_sph_in,p_idx_sph_in,g_str_sph,g_end_sph,dev_P1_sph,dev_SP1_sph,dev_P2_sph,dev_SP2_sph,dev_SP3_sph,
					dev_sort_storage_sph,&sort_storage_bytes_sph,
					g_idx_dem,p_idx_dem,g_idx_dem_in,p_idx_dem_in,g_str_dem,g_end_dem,dev_P1_dem,dev_SP1_dem,dev_P2_dem,dev_SP2_dem,dev_SP3_dem,
					dev_sort_storage_dem,&sort_storage_bytes_dem,
					file_P1_sph,file_P2_sph,file_P3_sph,
					file_P1_dem,file_P2_dem,file_P3_dem);
					
		}
		
		//-------------------------------------------------------------------------------------------------
		// Time-step Control
		//-------------------------------------------------------------------------------------------------
		if(tid==0){
			time+=dt;
			count++;

			//timestep is updated every 10 steps ------------ estimate new timestep (Goswami & Pajarola(2011))
			if((count%(freq_output/10))==0){
				dim3 t,b;
				t.x=128;
				// b.x=(num_part2-1)/t.x+1;
				// kernel_copy_max<<<b,t>>>(dev_P1,dev_SP3,max_rho,max_ft,max_umag);
				b.x=(num_part2_sph-1)/t.x+1;
				kernel_copy_max_sph<<<b,t>>>(dev_P1_sph,dev_SP3_sph,max_rho_sph,max_ft_sph,max_umag_sph);
				b.x=(num_part2_dem-1)/t.x+1;
				kernel_copy_max_dem<<<b,t>>>(dev_P1_dem,dev_SP3_dem,max_rho_dem,max_ft_dem,max_umag_dem);
				CUDA_CHECK(cudaDeviceSynchronize());

				// Find Max Velocity & Force using CUB - TID=0
				// cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_umag,d_max_umag0,num_part2);
				// cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_rho,d_max_rho0,num_part2);
				// cub::DeviceReduce::Max(dev_max_storage,max_storage_bytes,max_ft,d_max_ftotal0,num_part2);

				cub::DeviceReduce::Max(dev_max_storage_sph,max_storage_bytes_sph,max_umag_sph,d_max_umag0_sph,num_part_sph);
				cub::DeviceReduce::Max(dev_max_storage_sph,max_storage_bytes_sph,max_rho_sph,d_max_rho0_sph,num_part_sph);
				cub::DeviceReduce::Max(dev_max_storage_sph,max_storage_bytes_sph,max_ft_sph,d_max_ftotal0_sph,num_part_sph);

				cub::DeviceReduce::Max(dev_max_storage_dem,max_storage_bytes_dem,max_umag_dem,d_max_umag0_dem,num_part_dem);
				cub::DeviceReduce::Max(dev_max_storage_dem,max_storage_bytes_dem,max_rho_dem,d_max_rho0_dem,num_part_dem);
				cub::DeviceReduce::Max(dev_max_storage_dem,max_storage_bytes_dem,max_ft_dem,d_max_ftotal0_dem,num_part_dem);
				CUDA_CHECK(cudaDeviceSynchronize());
				
				// CUDA_CHECK(cudaMemcpy(max_umag0,d_max_umag0,sizeof(Real),cudaMemcpyDeviceToHost));
				// CUDA_CHECK(cudaMemcpy(max_rho0,d_max_rho0,sizeof(Real),cudaMemcpyDeviceToHost));
				// CUDA_CHECK(cudaMemcpy(max_ftotal0,d_max_ftotal0,sizeof(Real),cudaMemcpyDeviceToHost));

				CUDA_CHECK(cudaMemcpy(max_umag0_sph,d_max_umag0_sph,sizeof(Real),cudaMemcpyDeviceToHost));
				CUDA_CHECK(cudaMemcpy(max_rho0_sph,d_max_rho0_sph,sizeof(Real),cudaMemcpyDeviceToHost));
				CUDA_CHECK(cudaMemcpy(max_ftotal0_sph,d_max_ftotal0_sph,sizeof(Real),cudaMemcpyDeviceToHost));

				CUDA_CHECK(cudaMemcpy(max_umag0_dem,d_max_umag0_dem,sizeof(Real),cudaMemcpyDeviceToHost));
				CUDA_CHECK(cudaMemcpy(max_rho0_dem,d_max_rho0_dem,sizeof(Real),cudaMemcpyDeviceToHost));
				CUDA_CHECK(cudaMemcpy(max_ftotal0_dem,d_max_ftotal0_dem,sizeof(Real),cudaMemcpyDeviceToHost));


				// printf("%d\t rho_max=%5.2f\tu_max=%5.2f\tftotal_max=%5.2f\n\n",count,max_rho0[0],max_umag0[0],max_ftotal0[0]);
				printf("%d\t rho_max_sph=%5.2f\tu_max_sph=%5.2f\tftotal_max_sph=%5.2f\n\n",count,max_rho0_sph[0],max_umag0_sph[0],max_ftotal0_sph[0]);
				printf("%d\t rho_max_dem=%5.2f\tu_max_dem=%5.2f\tftotal_max_dem=%5.2f\n\n",count,max_rho0_dem[0],max_umag0_dem[0],max_ftotal0_dem[0]);
			}
		}

		pthread_barrier_wait(&barrier);
		if(time>=time_end) break;

	}


	//-------------------------------------------------------------------------------------------------
	// ##. Save Restart File
	//-------------------------------------------------------------------------------------------------

	if(ngpu==1) {

		save_restart(file_P1,file_P2,file_P3);
		cudaMemcpy(file_P1,dev_SP1,num_part2*sizeof(part1),cudaMemcpyDeviceToHost);
		cudaMemcpy(file_P2,dev_SP2,num_part2*sizeof(part2),cudaMemcpyDeviceToHost);
		cudaMemcpy(file_P3,dev_SP3,num_part2*sizeof(part3),cudaMemcpyDeviceToHost);


		free(file_P2);
		free(file_P3);
	}

	//-------------------------------------------------------------------------------------------------
	// ##. Memory Free
	//-------------------------------------------------------------------------------------------------
	free(file_P1);
	free(max_umag0);
	free(max_rho0);
	free(max_ftotal0);
	cudaFree(g_idx);
	cudaFree(p_idx);
	cudaFree(g_idx_in);
	cudaFree(p_idx_in);
	cudaFree(g_str);
	cudaFree(g_end);
	cudaFree(dev_P1);
	cudaFree(dev_SP1);
	cudaFree(dev_P2);
	cudaFree(dev_SP2);
	cudaFree(dev_SP3);
	cudaFree(max_umag);
	cudaFree(max_rho);
	cudaFree(max_ft);
	cudaFree(d_max_umag0);
	cudaFree(d_max_rho0);
	cudaFree(d_max_ftotal0);
	cudaFree(dev_sort_storage);
	cudaFree(dev_max_storage);
	cudaStreamDestroy(str1[tid]);
	cudaStreamDestroy(str2[tid]);
	pthread_barrier_wait(&barrier);

	return 0;
}


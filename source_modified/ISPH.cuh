#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err__), #call); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

void ISPH(int_t*vii,Real*vif)
{
	//-------------------------------------------------------------------------------------------------
	// GPU device properties
	//-------------------------------------------------------------------------------------------------
	struct cudaDeviceProp prop;
	{
		int_t gcount,i;
		cudaGetDeviceCount(&gcount);

		for(i=0;i<gcount;i++){
			cudaGetDeviceProperties(&prop,i);
			printf("### GPU DEVICE PROPERTIES.................................\n\n");
			printf("	Name: %s\n",prop.name);
			printf("	Compute capability: %d.%d\n",prop.major,prop.minor);
			printf("	Clock rate: %d\n",prop.clockRate);
			printf("	Total global memory: %ld\n",prop.totalGlobalMem);
			printf("	Total constant memory: %d\n",prop.totalConstMem);
			printf("	Multiprocessor count: %d\n",prop.multiProcessorCount);
			printf("	Shared mem per block: %d\n",prop.sharedMemPerBlock);
			printf("	Registers per block: %d\n",prop.regsPerBlock);
			printf("	Threads in warp: %d\n",prop.warpSize);
			printf("	Max threads per block: %d\n",prop.maxThreadsPerBlock);
			printf("	Max thread dimensions: %d,%d,%d\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
			printf("	Max grid dimensions: %d,%d,%d\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
			printf("...........................................................\n\n");
		}
	}
	printf(" ------------------------------------------------------------\n");
	printf(" SOPHIA_gpu v.1.0 \n");
	printf(" Developed by E.S. Kim,Y.B. Jo,S.H. Park\n");
	printf(" 2017. 02. 20 \n");
	printf(" Optimized by Y.W. Sim, CoCoLink Inc.\n");
	printf(" 2018, 2019(C) \n");
	printf(" Restructured & Innovated by Eung Soo Kim, Hee Sang Yoo, Young Beom Jo, Hae Yoon Choi, Su-San Park, Jin Woo Kim, Yelyn Ahn, Tae Soo Choi\n");
	printf(" ESLAB, SEOUL NATIONAL UNIVERSITY, SOUTH KOREA.\n");
	printf(" 2019. 08. 08 \n");
	printf("------------------------------------------------------------\n\n");
	//-------------------------------------------------------------------------------------------------


	//-------------------------------------------------------------------------------------------------
	// 입출력 파일 이름 설정 (초기화)
	//-------------------------------------------------------------------------------------------------

	char INPUT_FILE_NAME[128];
	strcpy(INPUT_FILE_NAME,"./input/input.txt");								// input file name and address


	//-------------------------------------------------------------------------------------------------
	// 입자 개수 세기
	//-------------------------------------------------------------------------------------------------

	num_part=gpu_count_particle_numbers2(INPUT_FILE_NAME);


	//-------------------------------------------------------------------------------------------------
	// 입력파일 읽고 host 입자 (HP1)에 정보 저장
	//-------------------------------------------------------------------------------------------------

	// host 입자 (HP1) 메모리 할당 & 초기화
	HP1=(part1*)malloc(num_part*sizeof(part1));
	memset(HP1,0,sizeof(part1)*num_part);


	// 입력파일 (input.txt) 읽기
	read_input(HP1);
	num_part_sph = gpu_count_particle_numbers_sph(HP1);	// SPH 입자의 개수
	num_part_dem = gpu_count_particle_numbers_dem(HP1);	// DEM 입자의 개수
	printf("Total number of SPH particles: %d\n",num_part_sph);	// sph 입자 개수 출력
	printf("Total number of DEM particles: %d\n",num_part_dem);	// dem
	HP1_sph=(part1*)malloc(num_part_sph*sizeof(part1));	// sph 입자 개수만큼 메모리 할당
	HP1_dem=(part1*)malloc(num_part_dem*sizeof(part1));	// dem 입자 개수만큼 메모리 할당
	memset(HP1_sph,0,sizeof(part1)*num_part_sph);	// sph 입자 정보 0으로 초기화
	memset(HP1_dem,0,sizeof(part1)*num_part_dem);	// dem 입자 정보 0으로 초기화
	read_sph_dem(HP1,HP1_sph,HP1_dem);	// sph 입자와 dem 입자의 정보를 HP1_sph와 HP1_dem에 저장


	//-------------------------------------------------------------------------------------------------
	// 셀 (Cell) 및 검색 범위 설정
	//-------------------------------------------------------------------------------------------------

	// 검색범위 관련 변수 (default)
	Real cell_reduction_factor=1.1;
	search_incr_factor=1.0;											// coefficient for cell and search range (esk)

	search_kappa=kappa;


	//-------------------------------------------------------------------------------------------------
	// 전체 계산 영역 파악 및 셀(Cell) 분할
	//-------------------------------------------------------------------------------------------------

	// 계산범위
	find_minmax(vii,vif,HP1);

	// 셀 간격(dcell)
	Real h0=HP1[0].h;
	dcell=cell_reduction_factor*kappa*h0;
	// Case 3-1-1 fixed bed domain: 200 mm x 40 mm x 40 mm.
	// The paper uses grid-based heat transfer; this case keeps the geometry but uses SPH interpolation.
	x_max = 0.210;
	x_min = -0.010;
	y_max = 0.045;
	y_min = -0.005;
	z_max = 0.045;
	z_min = -0.005;

  // 셀 개수(NI, NJ, NK)
	NI=(int)((x_max-x_min)/dcell)+1;
	NJ=(int)((y_max-y_min)/dcell)+1;
	NK=(int)((z_max-z_min)/dcell)+1;


		//-------------------------------------------------------------------------------------------------
	// GPU 당 셀 개수 및 필요한 입자 메모리 크기 계산
	//-------------------------------------------------------------------------------------------------

	// 각 GPU x 방향 셀 개수 (NI를 총 gpu 개수로 나눔)
	calc_area=ceil(NI/ngpu);

	// 각 GPU 당 입자 정보를 할당할 메모리 크기 설정
	if(ngpu>1){
		num_p2p=(int)(num_part*4/NI*C_p2p);						// (CAUTION)
		num_part2=(int)((num_part/ngpu)*1.2)+2*num_p2p;			// (CAUTION)
		num_part2_sph=(int)((num_part_sph/ngpu)*1.2)+2*num_p2p;	// (CAUTION)
		num_part2_dem=(int)((num_part_dem/ngpu)*1.2)+2*num_p2p;	// (CAUTION)
	}else{
		if (open_boundary)
		{
			space=h0*1.6;
			Nsx=(int)((x_max-x_min)/space)+1;		// (CAUTION)
			Nsy=(int)((y_max-y_min)/space)+1;		// (CAUTION)

			buffer_size=(Nsx*Nsy)*6;
		}

		//num_part2=num_part+buffer_size;
		num_part2=num_part;
		num_part2_sph=num_part_sph;
		num_part2_dem=num_part_dem;
	}
	printf("Number of particles per GPU: %d\n",num_part2);			// 각 GPU 당 입자 개수 출력
	printf("Number of SPH particles per GPU: %d\n",num_part2_sph);	// 각 GPU 당 SPH 입자 개수 출력
	printf("Number of DEM particles per GPU: %d\n",num_part2_dem);	// 각 GPU 당 DEM 입자 개수 출력


	//-------------------------------------------------------------------------------------------------
	// P2P 활성화
	//-------------------------------------------------------------------------------------------------


	if(ngpu>1){
		for(int i=0;i<ngpu;i++){
			cudaSetDevice(i);
			for(int j=0;j<ngpu;j++){
				cudaDeviceEnablePeerAccess(j,0);
			}
		}
	}


	//-------------------------------------------------------------------------------------------------
	// 코드 실행
	//-------------------------------------------------------------------------------------------------

	// WCSPH Dev Calc Function using Pthread
	pthread_t*solve_thread;
	void*thread_result;
	int*tid;

	tid=(int*)malloc(sizeof(int)*ngpu);
	solve_thread=(pthread_t*)malloc(sizeof(pthread_t)*ngpu);
	pthread_barrier_init(&barrier,NULL,ngpu);

	//WCSPH_Calc();
	for(int i=0;i<ngpu;i++){
		tid[i]=i;
		pthread_create(&solve_thread[i],NULL,ISPH_Calc,(void*)&tid[i]);
	}

	for(int i=0;i<ngpu;i++) pthread_join(solve_thread[i],&thread_result);
	pthread_barrier_destroy(&barrier);

	if(ngpu>1){
		for(int i=0;i<ngpu;i++){
			cudaSetDevice(i);
			for(int j=0;j<ngpu;j++){
				cudaDeviceDisablePeerAccess(j);
			}
		}
	}

	free(HP1);
	free(HP1_sph);
	free(HP1_dem);
	free(solve_thread);
	free(tid);
}

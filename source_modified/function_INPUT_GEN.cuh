#include <sys/stat.h>
#include <sys/types.h>

void save_input_dem(part1*P1_dem)
{
	// 저장 폴더 생성 (이미 있으면 무시됨)
	mkdir("./input_generated", 0777);

	// 파일명: 시뮬레이션 시간(초)을 포함하여 구분
	char fn[256];
	sprintf(fn, "./input_generated/input_%.2fs_%dstp.txt", time, count);

	FILE*outFile = fopen(fn, "w");
	if(outFile == NULL){
		printf("save_input_dem: cannot open %s\n", fn);
		return;
	}

	// header row (input_maker.py 의 index_list 와 동일)
	fprintf(outFile, "1\t2\t3\t7\t8\t9\t12\t27\t28\t29\n");

	// 입자 행 출력 (더미입자 제외: 실제 DEM 입자만 p_type>1000)
	int_t nop = num_part2_dem;
	int_t saved = 0;
	for(int_t i=0;i<nop;i++){
		if(P1_dem[i].p_type > 1000){
			fprintf(outFile,
				"%.6e\t%.6e\t%.6e\t%.6e\t%d\t%.6e\t%.6e\t%.6e\t%.6e\t%d\n",
				P1_dem[i].x,            // 1
				P1_dem[i].y,            // 2
				P1_dem[i].z,            // 3
				P1_dem[i].m,            // 7
				(int)P1_dem[i].p_type,  // 8
				P1_dem[i].h,            // 9
				P1_dem[i].rho,          // 12
				P1_dem[i].rad,          // 27
				P1_dem[i].ri,           // 28
				(int)P1_dem[i].dem_idx);// 29
			saved++;
		}
	}

	fclose(outFile);
	printf("save_input_dem: %s (%d particles)\n", fn, (int)saved);
}

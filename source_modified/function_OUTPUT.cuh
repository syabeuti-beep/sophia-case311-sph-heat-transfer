__global__ void kernel_copy_max_sph(part1*P1,part3*P3,Real*mrho,Real*mft,Real*mu)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].p_type==0||P1[i].i_type>=i_type_crt){
		mu[i]=0;
		mrho[i]=0;
		mft[i]=0;
		return;
	}

	mu[i]=sqrt(P1[i].ux*P1[i].ux+P1[i].uy*P1[i].uy+P1[i].uz*P1[i].uz);
	mrho[i]=P1[i].rho;
	mft[i]=P3[i].ftotal;
}

__global__ void kernel_copy_max_dem(part1*P1,part3*P3,Real*mrho,Real*mft,Real*mu)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_dem) return;
	if(P1[i].p_type==0||P1[i].i_type>=i_type_crt){
		mu[i]=0;
		mrho[i]=0;
		mft[i]=0;
		return;
	}

	mu[i]=sqrt(P1[i].ux*P1[i].ux+P1[i].uy*P1[i].uy+P1[i].uz*P1[i].uz);
	mrho[i]=P1[i].rho;
	mft[i]=P3[i].ftotal;
}

void save_restart(part1*P1,part2*P2,part3*P3)
{
	int_t i,nop;//,nob;
	nop=num_part;
	//int_t Nparticle=nop;									// number of fluid particles

	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	char FileName[256];
	sprintf(FileName,"./plotdata/restart.txt");
	// If the file already exists,its contents are discarded and create the new one.
	FILE*outFile;
	outFile=fopen(FileName,"w");

	fprintf(outFile,"1 2 3 4 5 6 7 8 9 10 11 12 13 26\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)

	//Write data -------------------------------------------------------------------------
	for(i=0;i<nop;i++){
			fprintf(outFile,"%f\t%f\t%f\t",P1[i].x,P1[i].y,P1[i].z);
			fprintf(outFile,"%f\t%f\t%f\t",P1[i].ux,P1[i].uy,P1[i].uz);
			fprintf(outFile,"%e\t%d\t%e\t",P1[i].m,P1[i].p_type,P1[i].h);
			fprintf(outFile,"%f\t%f\t%f\t",P1[i].temp,P1[i].pres,P1[i].rho);
			fprintf(outFile,"%f\t%f\n",P2[i].rho_ref,P1[i].flt_s);	//check f_total
			// fprintf(outFile,"%f\t%f\t%f\t",P2[i].rho_ref,P3[i].ftotal,P1[i].concn);	//check f_total
			// fprintf(outFile,"%f\t%f\t%d\t",P3[i].cc,P3[i].vis_t,P1[i].ct_boundary);
			// //fprintf(outFile,"%f\t%f\t%d\t%d\t",P3[i].cc,P3[i].vis_t,P1[i].ct_boundary,P3[i].hf_boundary);
			// fprintf(outFile,"%f\t%f\t%f\t%f\t%f\t%f\n",P3[i].lbl_surf,P3[i].drho,P3[i].denthalpy,P3[i].dconcn,P1[i].k_turb,P1[i].e_turb);
	}

	fclose(outFile);
}


float FloatSwap( float f )
{
   union
   {
      float f;
      unsigned char b[4];
      //unsigned char b[8];
   } dat1,dat2;

   dat1.f=f;
   dat2.b[0]=dat1.b[3];
   dat2.b[1]=dat1.b[2];
   dat2.b[2]=dat1.b[1];
   dat2.b[3]=dat1.b[0];
	 /*
   dat2.b[0]=dat1.b[7];
   dat2.b[1]=dat1.b[6];
   dat2.b[2]=dat1.b[5];
   dat2.b[3]=dat1.b[4];
   dat2.b[4]=dat1.b[3];
   dat2.b[5]=dat1.b[2];
   dat2.b[6]=dat1.b[1];
   dat2.b[7]=dat1.b[0];
	 //*/

   return dat2.f;
}

void save_plot_fluid_vtk_bin_solid_dem(part1*P1,part3*P3)
{
	int_t i,nop;//,nob;
	nop=num_part2_dem;
	// nob=number_of_boundaries;
	int_t Nparticle=0;							// number of fluid particles (x>0.00) for 3D PGSFR calculation
	//for(i=0;i<nop;i++) if(P1[i].x>0) Nparticle++;

	for(i=0;i<nop;i++) if(P1[i].dem_idx>0) Nparticle++;
	printf("test Particle %d\n",Nparticle);

	float val;
		float val1, val2, val3;
	int valt;

	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	char FileName_vtk[256];
	sprintf(FileName_vtk,"./plotdata/demDEM_decouple%d_dt%.0E_%dstp.vtk",decouple_stride,dt,count);
	// sprintf(FileName_vtk,"./plotdata/demDEM_decouple%d_dt%.2E_%dstp.vtk",decouple_stride,dt,count);
	// sprintf(FileName_vtk,"./plotdata/demDEM_decouple%d_%dstp.vtk",decouple_stride,count);
	// If the file already exists,its contents are discarded and create the new one.
	FILE*outFile_vtk;
	outFile_vtk=fopen(FileName_vtk,"w");

	fprintf(outFile_vtk,"# vtk DataFile Version 3.0\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
	fprintf(outFile_vtk,"Print out results in vtk format\n");			// header: description of file,it never exceeds 256 characters
	fprintf(outFile_vtk,"BINARY\n");														// format of data (ACSII / BINARY)
	fprintf(outFile_vtk,"DATASET POLYDATA\n");										// define DATASET format: 'POLYDATA' is proper to represent SPH particles

	//Define SPH particles---------------------------------------------------------------
	fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",Nparticle);					// define particles position as POINTS
	for(i=0;i<nop;i++){							// print out (x,y,z) coordinates of particles
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].x);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].y);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
			val=FloatSwap(P1[i].z);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"POINT_DATA\t%d\n",Nparticle);

	fprintf(outFile_vtk,"FIELD FieldData\t10\n");




	fprintf(outFile_vtk,"itype\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].i_type);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"mag_uijf\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].test6);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"Euler\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){	
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].eliz);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}







	

	fprintf(outFile_vtk,"porosity\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].DEMpor);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"uz\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].uz);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	



	fprintf(outFile_vtk,"Q_sd\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].Q_sd);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}
	




	fprintf(outFile_vtk,"ftotal_z\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P3[i].ftotalz);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}



	// fprintf(outFile_vtk,"ftotalx\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P3[i].ftotalx);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// } 



	fprintf(outFile_vtk,"temperature_surface\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].temp);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}




	// fprintf(outFile_vtk,"dtemp1_dt\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P3[i].dtemp1);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }





	fprintf(outFile_vtk,"rad	1	%d	float\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].rad);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"sph_solid_volume_fraction	1	%d	float\n",Nparticle);
	for(i=0;i<nop;i++){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].test1);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}

	fprintf(outFile_vtk,"sph_ks_eff	1	%d	float\n",Nparticle);
	for(i=0;i<nop;i++){
		if(P1[i].p_type>1000){
			val=FloatSwap(P1[i].test2);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}



	fprintf(outFile_vtk,"dtemp_dt	1	%d	float\n",Nparticle);
	for(i=0;i<nop;i++){
		//if(P1[i].x>0){
		if(P1[i].p_type>1000){
			val=FloatSwap(P3[i].dtemp);
			fwrite((void*)&val,sizeof(float),1,outFile_vtk);
		}
	}


	// fprintf(outFile_vtk,"mag_uij\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P1[i].pres);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"ftotaly\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P3[i].ftotaly);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	// 	fprintf(outFile_vtk,"i_type\t1\t%d\tint\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		valt=IntSwap(P1[i].i_type);
	// 		fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"demidx\t1\t%d\tint\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		valt=IntSwap(P1[i].dem_idx);
	// 		fwrite((void*)&valt,sizeof(int),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"flt_s\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P1[i].flt_s);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"flt_sd\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P1[i].flt_sd);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	// fprintf(outFile_vtk,"fdz_b\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type>1000){
	// 		val=FloatSwap(P1[i].Fdz_b);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }


	// fprintf(outFile_vtk,"curvature\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P3[i].curv);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	//
	// fprintf(outFile_vtk,"detection\t1\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val=FloatSwap(P3[i].lbl_surf);
	// 		fwrite((void*)&val,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	//
	// fprintf(outFile_vtk,"surface_tension\t3\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val1=FloatSwap(P3[i].fsx);
	// 		val2=FloatSwap(P3[i].fsy);
	// 		val3=FloatSwap(P3[i].fsz);
	// 		fwrite((void*)&val1,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val2,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val3,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	//
	// fprintf(outFile_vtk,"normal_vector_c\t3\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val1=FloatSwap(P3[i].nx_c);
	// 		val2=FloatSwap(P3[i].ny_c);
	// 		val3=FloatSwap(P3[i].nz_c);
	// 		fwrite((void*)&val1,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val2,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val3,sizeof(float),1,outFile_vtk);
	// 	}
	// }
	//
	// fprintf(outFile_vtk,"normal_vector\t3\t%d\tfloat\n",Nparticle);
	// for(i=0;i<nop;i++){
	// 	//if(P1[i].x>0){
	// 	if(P1[i].p_type==1){
	// 		val1=FloatSwap(P3[i].nx);
	// 		val2=FloatSwap(P3[i].ny);
	// 		val3=FloatSwap(P3[i].nz);
	// 		fwrite((void*)&val1,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val2,sizeof(float),1,outFile_vtk);
	// 		fwrite((void*)&val3,sizeof(float),1,outFile_vtk);
	// 	}
	// }

	fclose(outFile_vtk);
}

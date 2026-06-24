#define xx_box		0.04
#define yx_box 		0.0075
#define zx_box		0.125

#define Lxx_box 	0.08
#define Lyx_box  	100
#define Lzx_box 	0.25

#define rest_n 		0.97
#define rest_s 		-0.9
#define mu_s 		0.3
#define mu_rs		0.3

__global__ void KERNEL_treat_DEM_box_x_dem(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_dem) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;


	Real x,y,z;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real m,ri;

	Real F_box;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_box_mag;
	Real nx_box,ny_box,nz_box;
	Real sx_box,sy_box,sz_box;
	Real swx_box,swy_box,swz_box;

	Real x_center=xx_box;
	Real y_center=yx_box;
	Real z_center=zx_box;


	x=TP1[i].x;
	y=TP1[i].y;
	z=TP1[i].z;

	ux=TP1[i].ux;
	uy=TP1[i].uy;
	uz=TP1[i].uz;

	wx=TP1[i].wx;
	wy=TP1[i].wy;
	wz=TP1[i].wz;

	Real ftotalx,ftotaly,ftotalz;
	Real torqx,torqy,torqz;
	

	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	torqx=P3[i].torqx;
	torqy=P3[i].torqy;
	torqz=P3[i].torqz;



	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	F_box=fmax(fmax((abs(x-x_center)-(0.5*Lxx_box-rad)),(abs(y-y_center)-(0.5*Lyx_box-rad))),(abs(z-z_center)-(0.5*Lzx_box-rad)));

	if(F_box<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=x_center+fmin((0.5*Lxx_box-rad),fmax(-(0.5*Lxx_box-rad), x-x_center));
	cpy=y_center+fmin((0.5*Lyx_box-rad),fmax(-(0.5*Lyx_box-rad), y-y_center));
	cpz=z_center+fmin((0.5*Lzx_box-rad),fmax(-(0.5*Lzx_box-rad), z-z_center));

	

	//position update
	TP1[i].x=cpx;
	TP1[i].y=cpy;
	TP1[i].z=cpz;
	

	// sgn_x, sgn_y, sgn_z
	//Real sgn_x, sgn_y, sgn_z;
	Real sgn_x, sgn_y, sgn_z;

	sgn_x=(cpx-x)/(abs(cpx-x)+1e-20);
	sgn_y=(cpy-y)/(abs(cpy-y)+1e-20);
	sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	if (abs(cpx-x)<1e-12) sgn_x=0.0;
	if (abs(cpy-y)<1e-12) sgn_y=0.0;
	if (abs(cpz-z)<1e-12) sgn_z=0.0;

	//if (abs(ux)<1e-8) ux=0.0;
	//if (abs(uy)<1e-8) uy=0.0;
	//if (abs(uz)<1e-8) uz=0.0;
	//sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	//normal vector
	//nx_box=ux/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//ny_box=uy/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//nz_box=uz/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);

	nx_box=sgn_x/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	ny_box=sgn_y/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	nz_box=sgn_z/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);

	n_box_mag=sqrt(nx_box*nx_box+ny_box*ny_box+nz_box*nz_box);


	if(n_box_mag<0.5) return;		// actually do not contact the wall yet...

	un_mag=ux*nx_box+uy*ny_box+uz*nz_box;

	ucx=ux-rad*(wy*nz_box-wz*ny_box);
	ucy=uy-rad*(wz*nx_box-wx*nz_box);
	ucz=uz-rad*(wx*ny_box-wy*nx_box);

	ucx_w=-rad*(wy*nz_box-wz*ny_box);
	ucy_w=-rad*(wz*nx_box-wx*nz_box);
	ucz_w=-rad*(wx*ny_box-wy*nx_box);

	ucn_mag=ucx*nx_box+ucy*ny_box+ucz*nz_box;
	uc_wn_mag=ucx_w*nx_box+ucy_w*ny_box+ucz_w*nz_box;

	ucs_x=ucx-ucn_mag*nx_box;
	ucs_y=ucy-ucn_mag*ny_box;
	ucs_z=ucz-ucn_mag*nz_box;

	uc_ws_x=ucx_w-uc_wn_mag*nx_box;
	uc_ws_y=ucy_w-uc_wn_mag*ny_box;
	uc_ws_z=ucz_w-uc_wn_mag*nz_box;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_box=ucs_x/(ucs_mag+1e-20);
	sy_box=ucs_y/(ucs_mag+1e-20);
	sz_box=ucs_z/(ucs_mag+1e-20);

	swx_box=uc_ws_x/(uc_ws_mag+1e-20);
	swy_box=uc_ws_y/(uc_ws_mag+1e-20);
	swz_box=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt))) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sx_box;
		ffy=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sy_box;
		ffz=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sz_box;

		frfx=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swx_box;
		frfy=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swy_box;
		frfz=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swz_box;

		if ((ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx));

		P3[i].ftotalx=ftotalx;
		P3[i].ftotaly=ftotaly;
		P3[i].ftotalz=ftotalz;

		P3[i].torqx=torqx;
		P3[i].torqy=torqy;
		P3[i].torqz=torqz;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_box+uy_c*ny_box+uz_c*nz_box;

		TP1[i].ux=ux_c-u_cn_mag*nx_box;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uy=uy_c-u_cn_mag*ny_box;
		TP1[i].uz=uz_c-u_cn_mag*nz_box;

		TP1[i].wx=wx-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy))*k_dt;
		TP1[i].wy=wy-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz))*k_dt;
		TP1[i].wz=wz-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_box;
		Jny=-m*(1+rest_n)*ucn_mag*ny_box;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_box;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_box;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_box;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_box;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=ux+(1/m)*Jx;
		TP1[i].uy=uy+(1/m)*Jy;
		TP1[i].uz=uz+(1/m)*Jz;

		// angular velocity KERNEL_update
		TP1[i].wx=wx-(rad/ri)*(ny_box*Jz-nz_box*Jy);
		TP1[i].wy=wy-(rad/ri)*(nz_box*Jx-nx_box*Jz);
		TP1[i].wz=wz-(rad/ri)*(nx_box*Jy-ny_box*Jx);
	}
}

#define xy_box		0.04
#define yy_box 		0.0075
#define zy_box		0.125

#define Lxy_box 	100
#define Lyy_box  	0.015
#define Lzy_box 	100
__global__ void KERNEL_treat_DEM_box_y_dem(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_dem) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;


	Real x,y,z;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real m,ri;

	Real F_box;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_box_mag;
	Real nx_box,ny_box,nz_box;
	Real sx_box,sy_box,sz_box;
	Real swx_box,swy_box,swz_box;

	Real x_center=xy_box;
	Real y_center=yy_box;
	Real z_center=zy_box;


	x=TP1[i].x;
	y=TP1[i].y;
	z=TP1[i].z;

	ux=TP1[i].ux;
	uy=TP1[i].uy;
	uz=TP1[i].uz;

	wx=TP1[i].wx;
	wy=TP1[i].wy;
	wz=TP1[i].wz;

	Real ftotalx,ftotaly,ftotalz;
	Real torqx,torqy,torqz;
	

	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	torqx=P3[i].torqx;
	torqy=P3[i].torqy;
	torqz=P3[i].torqz;



	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	F_box=fmax(fmax((abs(x-x_center)-(0.5*Lxy_box-rad)),(abs(y-y_center)-(0.5*Lyy_box-rad))),(abs(z-z_center)-(0.5*Lzy_box-rad)));

	if(F_box<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=x_center+fmin((0.5*Lxy_box-rad),fmax(-(0.5*Lxy_box-rad), x-x_center));
	cpy=y_center+fmin((0.5*Lyy_box-rad),fmax(-(0.5*Lyy_box-rad), y-y_center));
	cpz=z_center+fmin((0.5*Lzy_box-rad),fmax(-(0.5*Lzy_box-rad), z-z_center));

	

	//position update
	TP1[i].x=cpx;
	TP1[i].y=cpy;
	TP1[i].z=cpz;
	

	// sgn_x, sgn_y, sgn_z
	//Real sgn_x, sgn_y, sgn_z;
	Real sgn_x, sgn_y, sgn_z;

	sgn_x=(cpx-x)/(abs(cpx-x)+1e-20);
	sgn_y=(cpy-y)/(abs(cpy-y)+1e-20);
	sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	if (abs(cpx-x)<1e-12) sgn_x=0.0;
	if (abs(cpy-y)<1e-12) sgn_y=0.0;
	if (abs(cpz-z)<1e-12) sgn_z=0.0;

	//if (abs(ux)<1e-8) ux=0.0;
	//if (abs(uy)<1e-8) uy=0.0;
	//if (abs(uz)<1e-8) uz=0.0;
	//sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	//normal vector
	//nx_box=ux/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//ny_box=uy/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//nz_box=uz/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);

	nx_box=sgn_x/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	ny_box=sgn_y/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	nz_box=sgn_z/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);

	n_box_mag=sqrt(nx_box*nx_box+ny_box*ny_box+nz_box*nz_box);


	if(n_box_mag<0.5) return;		// actually do not contact the wall yet...

	un_mag=ux*nx_box+uy*ny_box+uz*nz_box;

	ucx=ux-rad*(wy*nz_box-wz*ny_box);
	ucy=uy-rad*(wz*nx_box-wx*nz_box);
	ucz=uz-rad*(wx*ny_box-wy*nx_box);

	ucx_w=-rad*(wy*nz_box-wz*ny_box);
	ucy_w=-rad*(wz*nx_box-wx*nz_box);
	ucz_w=-rad*(wx*ny_box-wy*nx_box);

	ucn_mag=ucx*nx_box+ucy*ny_box+ucz*nz_box;
	uc_wn_mag=ucx_w*nx_box+ucy_w*ny_box+ucz_w*nz_box;

	ucs_x=ucx-ucn_mag*nx_box;
	ucs_y=ucy-ucn_mag*ny_box;
	ucs_z=ucz-ucn_mag*nz_box;

	uc_ws_x=ucx_w-uc_wn_mag*nx_box;
	uc_ws_y=ucy_w-uc_wn_mag*ny_box;
	uc_ws_z=ucz_w-uc_wn_mag*nz_box;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_box=ucs_x/(ucs_mag+1e-20);
	sy_box=ucs_y/(ucs_mag+1e-20);
	sz_box=ucs_z/(ucs_mag+1e-20);

	swx_box=uc_ws_x/(uc_ws_mag+1e-20);
	swy_box=uc_ws_y/(uc_ws_mag+1e-20);
	swz_box=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt)) ) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sx_box;
		ffy=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sy_box;
		ffz=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sz_box;

		frfx=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swx_box;
		frfy=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swy_box;
		frfz=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swz_box;

		if ((ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx));

		P3[i].ftotalx=ftotalx;
		P3[i].ftotaly=ftotaly;
		P3[i].ftotalz=ftotalz;

		P3[i].torqx=torqx;
		P3[i].torqy=torqy;
		P3[i].torqz=torqz;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_box+uy_c*ny_box+uz_c*nz_box;

		TP1[i].ux=ux_c-u_cn_mag*nx_box;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uy=uy_c-u_cn_mag*ny_box;
		TP1[i].uz=uz_c-u_cn_mag*nz_box;

		TP1[i].wx=wx-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy))*k_dt;
		TP1[i].wy=wy-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz))*k_dt;
		TP1[i].wz=wz-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_box;
		Jny=-m*(1+rest_n)*ucn_mag*ny_box;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_box;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_box;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_box;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_box;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=ux+(1/m)*Jx;
		TP1[i].uy=uy+(1/m)*Jy;
		TP1[i].uz=uz+(1/m)*Jz;

		// angular velocity KERNEL_update
		TP1[i].wx=wx-(rad/ri)*(ny_box*Jz-nz_box*Jy);
		TP1[i].wy=wy-(rad/ri)*(nz_box*Jx-nx_box*Jz);
		TP1[i].wz=wz-(rad/ri)*(nx_box*Jy-ny_box*Jx);
	}
}
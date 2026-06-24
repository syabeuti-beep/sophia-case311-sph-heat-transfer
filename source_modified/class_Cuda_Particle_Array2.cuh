// Cparticle Class Declaration
// Cparticle class contains particle information.
#ifndef max
#define max(a,b) (((a)>(b))?(a):(b))
#endif

#ifndef min
#define min(a,b) (((a)<(b))?(a):(b))
#endif
////////////////////////////////////////////////////////////////////////
typedef struct particles_array_1{
	uint_t i_type;													// Inner or Outer
	uint_t buffer_type;											// buffer_type (0: active , 1: inlet, 2: outlet)
	uint_t p_type;													// particle type: FLUID or BOUNDARY
	uint_t dem_idx;													// DEM particle index

	uint_t ct_boundary;												// const temperatrue particle index (const temp particle :1 / else : 0)

	Real x,y,z;															// (Predicted) positions [m] ( Predictor_Corrector : Predicted position / Euler : Real Position )
	Real x_star,y_star,z_star;
	Real ux,uy,uz;													// (Predicted) velocity [m/s] ( Predictor_Corrector : Predicted velocity / Euler : Real Velocity )
	Real wx,wy,wz;													// (Predicted) angular velocity [rad/s] ( Predictor_Corrector : Predicted angular velocity / Euler : Real angular Velocity )

	
	Real m;																	// mass [kg]
	Real ri;														// rotational inertia
	Real rad;														// radius of DEM part.
	Real h;																	// kernel distance
	Real temp, temp1, temp2, temp3;															// temperature [K]
	Real pres;															// pressure [Pa]
	//Real pres_ave;													// averaged pressure
	Real rho;																// density [kg/m3]	( Predictor_Corrector : Predicted density / Euler : Real density )

	Real flt_s;															// Shepard filter
	Real flt_sd;															// Shepard filter
	Real flt_sd_2;
	
	Real w_dx;															// w(dx) for particle shifting
	Real enthalpy;
	Real concn;

	Real grad_rhox,grad_rhoy,grad_rhoz;				// density gradient - only mass kernel using

	Real pgf_x,pgf_y,pgf_z;							// pressure gradient force term for DEM-SPH Coupling (force per unit volume)
	Real Fdx_b,Fdy_b,Fdz_b;								// SPH-DEM Coupling Force acting on DEM particle
	
	Real Fdx_df,Fdy_df,Fdz_df;								// SPH-DEM Coupling Force acting on DEM particle
	Real Fdx_da,Fdy_da,Fdz_da;								// SPH-DEM Coupling Force acting on DEM particle
	Real DEMpor;									// SPH particle porosity due to the DEM particle
	Real dDEMpor;									// porosity change rate at nth step
	Real dDEMpor_prev;								// porosity change rate at n-1th step
	Real DEMvf;										// DEM particle volume fraction sinked into the fluid
	Real Q_sd;
	Real Q_sdf;
	Real Q_f;

	Real ct_idx[16];												// Contact Particle index in DEM calculation
	Real del_s[16][3];												// Contact Particle shear overlap (accumulation) in DEM cal

	Real vol_power;


	// turbulence (by esk)
	Real k_turb,e_turb;												// turbulence kinetic energy,dissipation rate --> check unit
	Real pres_ipp;
	Real pres_ipp_p,pres_ipp_s;

	Real test1, test2, test3, test4, test5, test6, test7, test8;
	Real cond;
	Real vol0, vol;
	Real elix, eliy, eliz;
	//uint_t check;

	Real fbx, fby, fbz;
	Real fcx, fcy, fcz;
	int_t pos;
	Real PPE1, PPE2, PPE3, PPE4;
	Real XSPH_ux, XSPH_uy, XSPH_uz;	// XSPH velocity correction
	Real XSPH_temp;

	Real OpenBC_pres, OpenBC_rho, OpenBC_m, OpenBC_temp;
	Real OpenBC_ux, OpenBC_uy, OpenBC_uz;

	// Real nx_s,ny_s,nz_s;			// surface normal vector --> not used
}part1;
////////////////////////////////////////////////////////////////////////
typedef struct particles_array_2{
	Real rho_ref;

	//// turbulence (by esk)
	Real SR;																	// strain rate (2S:S)

	Real x0,y0,z0;													// Initial positions [m]
	Real ux0,uy0,uz0;												// Initial velocity [m/s]
	Real wx0,wy0,wz0;												// Initial angular velocity [rad/s]
	Real rho0;															// Initial density [kg/m3]
	Real drho0;														// Error compensation: divergence error
	Real rad0;

	Real vol0, dvol0;

	// psh: concentration diffusion
	Real concn0;														// concentration
	Real enthalpy0;													// enthalpy [J/kg]
	Real temp0, temp10, temp20, temp30;
}part2;
////////////////////////////////////////////////////////////////////////
typedef struct particles_array_3{

	
	Real drho;															// Time Derivative of density [kg/m3 s]
	Real dconcn;														// concentration time derivative
	Real denthalpy;
	Real drad;
	Real dtemp, dtemp1, dtemp2, dtemp3;
	Real torqx,torqy,torqz;								// total torq per rotational inertia (=angular acceleration [rad/s2] )
	Real ftotalx,ftotaly,ftotalz;						// total force [m/s2]
	Real ftotal;
	Real fpx,fpy,fpz;
	// turbulence (by esk)
	Real vis_t;																// turbulence viscosity
	Real Sxx,Sxy,Sxz,Syy,Syz,Szz;							// strain tensor... for SPH model
	Real dk_turb,de_turb;											// turbulence kinetic energy,dissipation rate --> check unit

	Real lbl_surf;													// surface label
	Real cc;																// color code

	Real Cm[Correction_Matrix_Size][Correction_Matrix_Size];
	
		// Real inv_cm_xx,inv_cm_yy,inv_cm_zz;
	// Real inv_cm_xy,inv_cm_yz,inv_cm_zx;

	Real nx,ny,nz;														// color code gradient for surface tension force (2016.09.02 jyb)
	Real nx_u,ny_u,nz_u;
	Real nx_w,ny_w,nz_w;													// unit wall normal pointing out of the wall (yhs)
	Real nx_t,ny_t,nz_t;													// wall tangential perpendicular to the contact line (yhs)
	Real nx_tl,ny_tl,nz_tl;													// new normal directions (yhs)

	Real nx_c,ny_c,nz_c;											// color code gradient for surface tension force (2016.09.02 jyb)
	Real nmag;																// 2017.04.20 jyb
	Real nmag_c;															// 2017.04.20 jyb
	Real curv;
	Real fsx,fsy,fsz;

	

	/*
	// psh:: ISPH info
	Real fpx,fpy,fpz;													// pressure force [m/s2]
	Real x_adv,y_adv,z_adv;										// predicted position by advection forces
	Real rho_err;															// difference between predicted density and reference density
	Real stiffness;														// stiffness parameter
	//----------------------------------------

	uint_t hf_boundary;											// heat flux particle index for heat transfer (heat flux particle : 1 / else : 0)
	Real cm_d; 															//--- not use
	Real p001;															// extra data	//--- not use
	Real curv; 															//--- not use
	Real num_density;												// particle number density [1/m3] //--- not use

	Real nd_ref;
	Real cm_xx,cm_yy,cm_zz;
	Real cm_xy,cm_yz,cm_zx;
	//*/


}part3;
///////////////////////////////////////////////////////////////////////////
typedef struct p2p_particles_array_3{
	Real drho;															// Time Derivative of density [kg/m3 s]
	Real dconcn;														// concentration time derivative
	Real denthalpy;
	Real torqx,torqy,torqz;
	Real ftotalx,ftotaly,ftotalz;						// total force [m/s2]
	Real ftotal;
}p2p_part3;

#define		cu_memset				0xffffffff
#define 	vii_size				64
#define 	vif_size				32
#define		Max_GPU					10

#define 	Wcsph 					0
#define 	Isph 					1
#define 	Dfsph 					2

#define 	FLUID					1
#define 	BOUNDARY				0
#define 	MOVING					9
#define 	CORIUM					50

#define 	CONCRETE				60
#define 	CONCRETE_SOL	    	-60
#define 	MCCI_CORIUM		    	65

#define 	IVR_METAL				70
#define 	IVR_CORIUM			    75
#define 	IVR_VESSEL			    80
#define 	IVR_VESSEL_SOL      	-80

#define     DEM_SOLID               1001

#define 	DUMMY_IN				100
#define 	Y_IN					1

#define 	Liquid 					0
#define 	Gas 					1
#define 	Solid 					2

#define 	Single_Phase		    1
#define 	Two_Phase				2

#define 	WRONG_INDEX		    	1e8			// Limitation of WRONG_INDEX: 2.2e9 (signed integer)

#define 	Gaussian				0
#define 	Quintic					1
#define 	Quartic					2
#define 	Wendland2				3
#define 	Wendland4				4
#define 	Wendland6				5
#define   Cubic4  6

#define Cosine  0
#define Gauss   1
#define Modified_Gaussian   2   //for IPF
#define Cubic   3
#define Wend2   4

#define DENSITY_WATER   1000.0
#define DENSITY_AIR   1.2       // Case 3-1-1 gas density from Imatani & Sakai (2025), Table 1

#define VISCOSITY_WATER 1.0e-3
#define VISCOSITY_AIR   2.0e-5

//// open boudary /////
#define Inlet 1
#define Outlet 2
#define Left 3
#define Right 4
#define Inlet_Density 1.2
#define Inlet_Velocity 1.0
#define Inlet_Temp 293.15

#define 	Mass_Sum				0				// Mass_Summation Method
#define 	Continuity		    	1				// Continuity Equation Method

#define 	Shepard					0				// Shepard Filter
#define 	MLS						1				// Moving Least Square(MLS) Filter

#define 	Euler					0				// Euler Explicit Time Stepping
#define 	Pre_Cor					1				// Predictor-Corrector Time Stepping

#define 	Potential				1				// Surface tension force based on inter-particle potential energy (Single / Two phase)
#define 	Curvature				2				// Surface tension force based on surface curvature ( Two phase )

#define 	PI						3.141592653

#define 	Gravitational_CONST     9.800	// gravitational constant [m/s2]

#define 	Alpha					0.1		// coefficient alpha of artificial viscosity
#define 	Beta					0.02		// coefficient beta of artificial viscosity

#define 	NORMAL_THRESHOLD 		0.001			// normal threshold for surface tension force


#define		Water					0
#define		Metal					1

#define 	epsilon					1e-6		// denominator
#define 	delta					0.1			// delta-SPH coefficient
#define 	K_repulsive		    	0.0001	// constant for repulsive boundary force

#define 	C_mu					0.09
#define 	C_e1					1.44
#define 	C_e2					1.92
#define 	sigma_k					1.0
#define 	sigma_e					1.3
#define 	kappa_t					0.41
#define 	Cs_SPS					0.12			// check please (by esk)
#define 	CI_SPS					0.00066
#define 	L_SPS					0.01			// scale of length scale (for test)

#define 	Lm						0.01
#define 	Maximum_turbulence_viscosity	0.1
#define 	Laminar					0
#define 	K_LM					1
#define 	K_E						2
#define 	SPS						3
#define 	HB      				4

#define 	DIFF_DENSITY		    0
#define   Correction_Matrix_Size  3

#define 	KGC		    1
#define 	FPM		    2
#define 	DFPM		  3
#define 	KGF		    4

#define   Molteni   1
#define   Antuono   2

#define   Min_det   1e-4      // minimum of determinent

#define   Pb    0.0
#define   i_type_crt  2
#define   C_p2p   1.5

#define table_size 50


#define s_ff1    1.1e-6
#define s_ff2    0
#define s_f1f2   0
#define s_sf1    0
#define s_sf2    0
#define s_s2f1   0
#define s_s2f2   0

#define A_ff1    0
#define A_ff2    0
#define A_f1f2   0
#define A_sf1    0
#define A_sf2    0
#define A_s2f1    0
#define A_s2f2    0

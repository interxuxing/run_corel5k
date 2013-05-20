/*
 * =============================================================
 * gEloss.c 
  
 * takes six inputs,  old_gradient (1xd)
 *					  D_target (1xp)
 *                    D_imposter (1xn)
 *					  V_target (pxd)
 *					  V_imposter (nxd)
 * 					  \mu
 * 
 * gEloss calculate gradient for the q-th image as in [1], where
 *
 *		Eloss = (1-\mu) \sum_{p->q}D(I_q, I_p) + \mu \sum_{p->q, n}
 *				(1 - y_{qn}) Hingeloss{1 + D(I_q, I_p) - D(I_q, I_n)} 
 *
 *
 *		gEloss = (1-\mu) \sum_{p->q}d(I_q, I_p)\mu + \sum_{p->q, n}
 *				(1 - y_{qn}) {d(I_q, I_p) - d(I_q, I_n)}
 *
 * =============================================================
 */
 
 
 
#include "mex.h"

/* If you are using a compiler that equates NaN to zero, you must
 * compile this example using the flag -DNAN_EQUALS_ZERO. For 
 * example:
 *
 *     mex -DNAN_EQUALS_ZERO gEloss.c  
 *
 * This will correctly define the IsNonZero macro for your
   compiler. */

#if NAN_EQUALS_ZERO
#define IsNonZero(d) ((d) != 0.0 || mxIsNaN(d))
#else
#define IsNonZero(d) ((d) != 0.0)
#endif

#define OLD_G_IN prhs[0]
#define D_TARGET_IN prhs[1]
#define D_IMPOSTER_IN prhs[2]
#define V_TARGET_IN prhs[3]
#define V_IMPOSTER_IN prhs[4] 
#define MU_IN prhs[5]


#define Eloss_OUT plhs[0]
#define gEloss_OUT plhs[1]  
  
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
   double mu, mu_target, mu_imposter;
   double dEloss = 0, dHingeloss = 0;
   int i,j,k, dim_target, dim_imposter, num_target, num_imposter;
   double *ptr_gEloss, *ptr_Eloss, *ptr_target_D, 
			*ptr_target_V, *ptr_imposter_D, *ptr_imposter_V;
   
/* Check for proper number of input and output arguments. */    
  if (nrhs != 6) {
    mexErrMsgTxt("Six input arguments required.");
  } 
  if (nlhs > 2) {
    mexErrMsgTxt("Too many output arguments.");
  }
  
  /* Check data type of input argument. */
  if (!(mxIsDouble(OLD_G_IN)) && !(mxIsDouble(D_TARGET_IN)) && 
		!(mxIsDouble(D_IMPOSTER_IN)) && !(mxIsDouble(V_TARGET_IN)) && 
		!(mxIsDouble(V_IMPOSTER_IN))) {
	mexErrMsgTxt("Input arrays must be of type double.");
  }
  mu = mxGetScalar(MU_IN);
  if (mu >= 1 || mu <= 0){
	mexErrMsgTxt("mu should be a scalar between 0 and 1 !");
  }
  
  // get dim of target and imposter samples
  dim_target = mxGetN(V_TARGET_IN);
  dim_imposter = mxGetN(V_IMPOSTER_IN);
  
  if (dim_target != dim_imposter){
	mexErrMsgTxt(" Dim of target and imposter should be the same!\n");
  }
  // get numbers of target and imposter samples
  num_target = mxGetM(V_TARGET_IN);
  num_imposter = mxGetM(V_IMPOSTER_IN);

  // create memory for output gEloss
  gEloss_OUT = mxCreateDoubleMatrix(1, dim_target, mxREAL);
  ptr_gEloss = (double *)mxGetPr(gEloss_OUT);
  
  // create memory for output dEloss
  Eloss_OUT = mxCreateDoubleScalar(0);
  ptr_Eloss = (double *)mxGetPr(Eloss_OUT);
  
  // now loop to calculate the gradient of Eloss
  // the total loop times is num_target*num_imposter
  mu_target = 1 - mu;
  mu_imposter = mu;
  
  ptr_target_D = (double *)mxGetPr(D_TARGET_IN);
  ptr_imposter_D = (double *)mxGetPr(D_IMPOSTER_IN);
  
  ptr_target_V = (double *)mxGetPr(V_TARGET_IN);
  ptr_imposter_V = (double *)mxGetPr(V_IMPOSTER_IN);
 
  #if MEX_DEBUG
   mexPrintf("------- Now caculate the loss and gradient for given tripples: \n");
   mexPrintf("-----Target samples: %d, imposter samples: %d, with dimension %d \n\n", 
			num_target, num_imposter, dim_target);
  #endif
  // calculate loss
  for (i = 0; i < num_target; i++){
	// add the first part (eloss)
	*ptr_Eloss += mu_target * ptr_target_D[i];
	for (j = 0; j < num_imposter; j++){
		dHingeloss = 1 + ptr_target_D[i] - ptr_imposter_D[j];
		if(dHingeloss > 0){
			*ptr_Eloss += mu_imposter * dHingeloss;
		}
	}
  }
  
  // mexPrintf("calculate loss value finished! \n");
  
  // calculate gradient
	  for (i = 0; i < num_target; i++){
		
		for (k = 0; k < dim_target; k++){
			// add the first part (gradient)
			ptr_gEloss[k] += mu_target * ptr_target_V[i*dim_target + k];
			
			for (j = 0; j < num_imposter; j++){
				
					dHingeloss = 1 + ptr_target_D[i] - ptr_imposter_D[j];
					// add the second part, if hingeloss > 0
					if (dHingeloss > 0)
						ptr_gEloss[k] += mu_imposter * (ptr_target_V[i*dim_target + k] - 
							ptr_imposter_V[j*dim_imposter + k]);
		}
	  }
  }
  
  // mexPrintf("calculate gradient finished! \n");
}
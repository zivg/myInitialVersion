/* ----------------------------------------------------------------------
   LIGGGHTS - LAMMPS Improved for General Granular and Granular Heat
   Transfer Simulations

   LIGGGHTS is part of the CFDEMproject
   www.liggghts.com | www.cfdem.com

   Stefan Radl, radl@tugraz.at
   Copyright 2013  Graz University of Technology

   LIGGGHTS is based on LAMMPS
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   This software is distributed under the GNU General Public License.

   See the README file in the top-level directory.

------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------
   Contributing authors:
   Stefan Radl (TU Graz)
   Bhageshvar Mohan (TU Graz)

   Description
    Fix for liquid transfer among particles with models A,B1,B2 and C

      Model A  - Simple conduction based transfer rate model
      Model B1 - Instantaneous liquid transfer model based on particle-particle surface contact
      Model B2 - Instantaneous liquid transfer model considering film thickness and rupture distance
      Model C  - Filling rate based model for the drainage of liquid into the bridge
------------------------------------------------------------------------- */

#include "fix_liquidtracking_instant.h"
#include "atom.h"
#include "compute_pair_gran_local.h"
#include "fix_property_atom.h"
#include "fix_property_global.h"
#include "force.h"
#include "math_extra.h"
#include "mech_param_gran.h"
#include "modify.h"
#include "neigh_list.h"
#include "pair_gran.h"
#include "math.h"
#include "string.h"
#include "domain.h"
#include "comm.h"
#include "update.h"
#include "error.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "respa.h"
#include "math_const.h"
#include "pointers.h"


#if defined(_WIN32) || defined(_WIN64)
#include <algorithm>
#define fmax std::max
#define fmin std::min
#endif

#include "stdlib.h"
#include "random_park.h"
#include "contact_models.h"

 ////////////////////////////////////////////////////////////////////////////////////////////////////////////
// make sure that there are no duplicate call here. "mpi.h" and "stdio.h" are also read by "contact_models.h"
// check whether those two includes can be removed.
#include "mpi.h"
#include "stdio.h"
#include "output.h"
#include "dump_dcd.h"
#include <vector>

#include <list>
#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>      // std::stringstream
#include <fstream>


 ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;


/*-------------------------------------------------------------------------*/

void updateConnectivity(int & i, int &j,int & currentProcessor, FILE *connectivityFile);
void printNoOfParticlesOnCPU(int currentCPU, int adtSize);


/*-------------------------------------------------------------------------*/

/* ---------------------------------------------------------------------- */

FixLiquidTrackingInstant::FixLiquidTrackingInstant(class LAMMPS *lmp, int narg, char **arg) : FixLiquidTracking(lmp, narg, arg)
{
  int iarg = 5;

  nevery = 1;
  area_correction_flag = 0;
  model_switch_flag    = 0;

//Parameters used for Model C
  liquidViscosity = 0.001;    //Dynamic Viscosity of liquid (kg/s.m) or (Pa.s)
  surfaceTension  = 0.001;    //Surface tension of liquid (kg/s^2) or (N/m)
  t_star          = 1.0;      //T*, dimensionless time value (-)
  v_init_star     = 0;        //Vinit* value (initial liquid bridge volume)
  a               = 1;        //Dimensionless filling rate coefficient (-)
  endOfStep       = 0;        //For Model - C4 (future work)
  modelC_switch   = 0;        //To switch between various submodels in Model C
  f_mobile        = 13.33e-2; //Maximum fraction of liquid mobile to flow into the bridge
  transfer_ratio  = 0.5;
  bool hasargs    = true;
  seed            = 99;       //Seed value for random value generator
  areaModel       = 0;        //Switch between methods to calculate fractional area in surface coverage models
  roughnessLength = 0.0;      //Dimensionless Surface roughness length for roughness submodel
  k_constant      = 1.65;     //Empirical constant for simple surface coverage model

//To Switch between models (A,B1,B2,C)

  while(iarg < narg && hasargs)
  {
    hasargs = false;
    if(strcmp(arg[iarg],"model_switch") == 0) {
      if (iarg+2 > narg) error->fix_error(FLERR,this,"not enough arguments for keyword 'model_switch'");
      if(strcmp(arg[iarg+1],"A") == 0)
        model_switch_flag = 0;
      else if(strcmp(arg[iarg+1],"B1") == 0)
        model_switch_flag = 1;
      else if(strcmp(arg[iarg+1],"B2") == 0)
        model_switch_flag = 2;
      else if(strcmp(arg[iarg+1],"C") == 0)
        model_switch_flag = 3;
      else error->fix_error(FLERR,this,"");
      iarg += 2;
      hasargs = true;
    } else if (strcmp(arg[iarg],"liquid_prop") == 0) {
      if (iarg+3 > narg) error->fix_error(FLERR,this,"not enough arguments for keyword 'liquid_prop'");
        liquidViscosity = atof(arg[iarg+1]);
        surfaceTension = atof(arg[iarg+2]);
      if ((liquidViscosity <= 0.0)||(surfaceTension <= 0.0)) error->all(FLERR,"Illegal liquid_prop value. Please set liquid_prop value >0");
      iarg +=3;
      hasargs = true;
    } else if (strcmp(arg[iarg],"NumParams") == 0) {
      if (iarg+5 > narg) error->fix_error(FLERR,this,"not enough arguments for keyword 'NumParams'");
        t_star = atof(arg[iarg+1]);
        v_init_star = atof(arg[iarg+2]);
        a      = atof(arg[iarg+3]);
        f_mobile      = atof(arg[iarg+4]);
      if ((t_star <= 0.0)||(v_init_star < 0.0)||(a <= 0.0)||(f_mobile < 0.0)) error->all(FLERR,"Illegal values for numerical parameters. Please set t_star and 'a' value >0. v_init_star and f_mobile value >=0");
      iarg +=5;
      hasargs = true;
    } else if (strcmp(arg[iarg],"endOfStep") == 0) {
      if (iarg+2 > narg) error->fix_error(FLERR,this,"not enough arguments for keyword 'endOfStep'");
      if(strcmp(arg[iarg+1],"yes") == 0)
      endOfStep = 1;
      else if(strcmp(arg[iarg+1],"no") == 0)
      endOfStep = 0;
      else error->all(FLERR,"Illegal endOfStep option called");
      iarg +=2;
      hasargs = true;
    } else if (strcmp(arg[iarg],"modelC_switch") == 0) {
      if (iarg+2 > narg) error->fix_error(FLERR,this,"not enough arguments for keyword 'modelC_switch'");
      if(strcmp(arg[iarg+1],"C1") == 0)
      modelC_switch = 0;
      else if(strcmp(arg[iarg+1],"C2") == 0)
      modelC_switch = 1;
      else if(strcmp(arg[iarg+1],"C3") == 0)
      modelC_switch = 2;
      else if(strcmp(arg[iarg+1],"C4") == 0)
      modelC_switch = 3;  
      else error->all(FLERR,"Illegal modelC_switch option called. To switch between C1,C2,C3 and C4");
      iarg +=2;
      hasargs = true;
    } else if (strcmp(arg[iarg],"transfer_ratio") == 0) {
      if (iarg+2 > narg) error->fix_error(FLERR,this,"not enough arguments for keyword 'transfer_ratio'");
      transfer_ratio = atof(arg[iarg+1]);
      if ((transfer_ratio <= 0.0)) error->all(FLERR,"Illegal values for transfer_ratio. Please set value higher than 0");
      iarg +=2;
      hasargs = true;

    } else if (strcmp(arg[iarg],"roughness") == 0) {
      if (iarg+2 > narg) error->fix_error(FLERR,this,"not enough arguments for keyword 'roughnessLength'");
      roughnessLength = atof(arg[iarg+1]);
      if ((roughnessLength < 0.0)) error->all(FLERR,"Illegal values for roughnessLength. Please set value 0 or greater");
      iarg +=2;
      hasargs = true;

    } else if (strcmp(arg[iarg],"surfaceCov") == 0) {
      if (iarg+4 > narg) error->fix_error(FLERR,this,"not enough arguments for keyword 'surfaceCov'");
      areaModel = atof(arg[iarg+1]);
      k_constant = atof(arg[iarg+2]);
      n_patch = atof(arg[iarg+3]);
      if ((areaModel < 0.0)||(k_constant < 0.0)||(n_patch < 0.0)) error->all(FLERR,"Illegal values for surfaceCov. Please set value 0 or greater");
      iarg +=4;
      hasargs = true;

    } else if(strcmp(style,"liquidtracking/instant") == 0)
      	error->fix_error(FLERR,this,"unknown keyword");
  }

  fix_transferCoefficient = NULL;
  transferCoefficient     = NULL;
//   random number generator, different for CPUs
  random_equal = new RanPark(lmp,seed + 3000 * comm->me);

}

/* ---------------------------------------------------------------------- */

FixLiquidTrackingInstant::~FixLiquidTrackingInstant()
{

	if (transferCoefficient)
		delete []transferCoefficient;
  delete random_equal;

}

/* ---------------------------------------------------------------------- */

// post_create() of parent is fine

/* ---------------------------------------------------------------------- */

void FixLiquidTrackingInstant::pre_delete(bool unfixflag)
{

  // tell cpl that this fix is deleted
  if(cpl && unfixflag) cpl->reference_deleted();

}

/* ---------------------------------------------------------------------- */

int FixLiquidTrackingInstant::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= INITIAL_INTEGRATE;
//  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */
int FixLiquidTrackingInstant::n_history_extra()
{

  if (model_switch_flag != 0)
  {
    extra_value = 1;
  }
  if (modelC_switch == 2)
  {
    extra_value = 4;  
  }
  return extra_value;
}

/* ---------------------------------------------------------------------- */
bool FixLiquidTrackingInstant::history_args(char** args)
{
    //provide names and newtonflags for each history value
    //newtonflag = 0 means that the value is same

    if (model_switch_flag != 0)
    {
      args[0] = (char *) "touchliquid";
      args[1] = (char *) "0";
    }
    if (modelC_switch == 2)
    {
      args[2] = (char *) "counter";
      args[3] = (char *) "0";
      args[4] = (char *) "counter_wet_i";
      args[5] = (char *) "0";
      args[6] = (char *) "counter_wet_j";
      args[7] = (char *) "0";
    }
    return true;
}
/* ---------------------------------------------------------------------- */

void FixLiquidTrackingInstant::init()
{

  if (FHG_init_flag == false){
    FixLiquidTracking::init();
  }
  double expo;
  int max_type = pair_gran->mpg->max_type();
  dnum = pair_gran->dnum_pair();
  dnum_mine = pair_gran->fix_extra_dnum_index(this);

  if (transferCoefficient) delete []transferCoefficient;
  transferCoefficient = new double[max_type];
  fix_transferCoefficient = static_cast<FixPropertyGlobal*>(modify->find_fix_property("liquidTransferCoefficient","property/global","scalar",max_type,0,style));

  // pre-calculate conductivity for possible contact material combinations
  for(int i = 1; i < max_type + 1; i++)
      for(int j = 1; j < max_type + 1; j++)
      {
          transferCoefficient[i-1] = fix_transferCoefficient->compute_vector(i-1);
          if(transferCoefficient[i-1] < 0.) error->all(FLERR,"Fix liquidtracking/instant: Liquid transferCoefficient must not be < 0");
      }

  if(area_correction_flag)
  {
    if(force->pair_match("gran/hooke",0)) expo = 1.;
    else if(force->pair_match("gran/hertz",0)) expo = 2./3.;
    else error->fix_error(FLERR,this,"area correction could not identify the granular pair style you are using, supported are hooke and hertz types");

    FixPropertyGlobal* ymo = static_cast<FixPropertyGlobal*>(modify->find_fix_property("youngsModulusOriginal","property/global","peratomtype",max_type,0,style));

    const double * Y      = static_cast<FixPropertyGlobal*>(modify->find_fix_property("youngsModulus","property/global","peratomtype",max_type,0,style))->get_values();
    const double * nu     = static_cast<FixPropertyGlobal*>(modify->find_fix_property("poissonsRatio","property/global","peratomtype",max_type,0,style))->get_values();
    const double * Y_orig = ymo->get_values();

    // allocate a new array within youngsModulusOriginal
    ymo->new_array(max_type,max_type);

    // feed deltan_ratio into this array
    for(int i = 1; i < max_type+1; i++)
    {
      for(int j = 1; j < max_type+1; j++)
      {
        double Yeff_ij      = 1./((1.-pow(nu[i-1],2.))/Y[i-1]     +(1.-pow(nu[j-1],2.))/Y[j-1]);
        double Yeff_orig_ij = 1./((1.-pow(nu[i-1],2.))/Y_orig[i-1]+(1.-pow(nu[j-1],2.))/Y_orig[j-1]);
        double ratio = pow(Yeff_ij/Yeff_orig_ij,expo);

        ymo->array_modify(i-1,j-1,ratio);
      }
    }

    // get reference to deltan_ratio
    deltan_ratio = ymo->get_array_modified();
  }

  updatePtrs();
}

/* ---------------------------------------------------------------------- */

void FixLiquidTrackingInstant::initial_integrate(int vflag)
{

  updatePtrs();

  //reset liquid flux
  //sources are not reset
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
  {
     if (mask[i] & groupbit)
     {
      condLiqFlux[i][0]     = 0.;
      condLiqFlux[i][1]     = 0.;
      condLiqFlux[i][2]     = 0.;
      liqInBridge[i][0]     = 0.;
      surfaceCoverage[i][0] = 0.;
      BridgeNo[i][0]        = 0.;
//        BridgeTime[i][0]      = 0.;
     }
  }

  //update ghosts

  fix_condLiqFlux->do_forward_comm();
  fix_liqInBridge->do_forward_comm();
  fix_surfaceCoverage->do_forward_comm();
  fix_BridgeNo->do_forward_comm();
// fix_BridgeTime->do_forward_comm();

}

/* ---------------------------------------------------------------------- */

void FixLiquidTrackingInstant::post_force(int vflag){

  //template function for using touchflag or not
  if(history_flag == 0) post_force_eval<0>(vflag,0);
  if(history_flag == 1) post_force_eval<1>(vflag,0);

}

/* ---------------------------------------------------------------------- */

void FixLiquidTrackingInstant::cpl_evaluate(ComputePairGranLocal *caller)
{
  if(caller != cpl) error->all(FLERR,"Illegal situation in FixLiquidTrackingInstant::cpl_evaluate");
  if(history_flag == 0) post_force_eval<0>(0,1);
  if(history_flag == 1) post_force_eval<1>(0,1);
}

/* ---------------------------------------------------------------------- */

template <int HISTFLAG>
void FixLiquidTrackingInstant::post_force_eval(int vflag,int cpl_flag)
{

  double condFlux[3], condFlux_i[3], condFlux_j[3], flux(0.0), flux_i(0.0), flux_j(0.0);
  int *touch, *touchpair;
  double mbridge_i(0.0), mbridge_j(0.0), n_wet(0.0), peff_i(0.0),peff_j(0.0);
  int n_bridge(0);
  bool liquid_exchange;

  int newton_pair = force->newton_pair;
  int inum = pair_gran->list->inum;
  int * ilist = pair_gran->list->ilist;
  int * numneigh = pair_gran->list->numneigh;
  int ** firstneigh = pair_gran->list->firstneigh;
  int ** firsttouch = pair_gran->list->listgranhistory->firstneigh;
  double ** firsthist = pair_gran->list->listgranhistory->firstdouble;

  double *radius = atom->radius;
  double *rmass = atom->rmass;
  double **x = atom->x;
  int nlocal = atom->nlocal;

  int *mask = atom->mask;

  double dt = update->dt;

  double fourThirdPi = 4.0/3.0 * MY_PI;
  double invFourThirdPi = 1.0 / fourThirdPi;
  double oneByThree = 1.0/3.0 ;
  double skinDist = neighbor->skin;
  updatePtrs();

//////////////////////////////////////////////////////////////////////////////////////////////// Ziv
  int *tag = atom->tag; // Create global particle ID array
  int natoms = static_cast<int> (lmp->atom->natoms); // getting the total number of particles //printf("natoms = %d",natoms);
  int adtSize;

  int me;
  int nprocs;
  int MAX_LIQUID_BRIDGES=15;
  vector <int> noOfParticlesOnCPUs; // a holder for all the inums

  MPI_Comm_rank(world,&me); // Getting the current rank
  MPI_Comm_size(world,&nprocs);
  MPI_Status status;

  FILE * connectivityFile;
  char filename[128];
  char connectivityFilename [128];

  bool printTimeStep=false;
  //bool SEPARATED_CONNECTIVITY_DIRS = false;// This flag when set true would create a a connectivity file for each timestep for each CPU.
  bool GENERATE_INTEGRATED_CONNECTIVITY_DIRS = true; // This flag when set true would create a reduced single connectivity file for each timestep.

  char timePathName[128];
  char cpuDirectoryPathName[128];
  char str1[128];
  char str2[128];
  char str3[128];

  int currentTimeStep=update->ntimestep;
  int printFrequency=output->thermo_every;// The Connectivity data will be collected according to the output rate of fix_thermo.

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  noOfParticlesOnCPUs.resize(nprocs);

  int nparticles,nparticles_tot;

       nparticles = atom->nlocal;
       nparticles_tot = static_cast<int> (atom->natoms);

       printf("I'm on CPU %d. local particles are %d, total number of particles are : %d \n", me,nparticles,nparticles_tot);


  if (me==0) // this part is used to send the particles sizes CHECK IF THERE IS A NEED TO LOOP OVER ALL PROCESSORS
  {
	  int received_CPUnoOfParticles;
      int p;


      noOfParticlesOnCPUs[0]=inum;
      for(int p=1;p<nprocs;p++)
      {
    	  MPI_Recv(&noOfParticlesOnCPUs[p], 1, MPI_INT, p, MPI_ANY_TAG, world, &status);
    	  //noOfParticlesOnCPUs[p]=received_CPUnoOfParticles;
    	  printf("I got size of data from CPU %d which is : %d\n", p, noOfParticlesOnCPUs[p]);

      }

      printf("the particles division over processors is : \n");
      int psum=0;
      for(int p=0;p<nprocs;p++)
           {
    	      printf("cpu %d , No. of particles %d\n",p,noOfParticlesOnCPUs[p]);
    	      psum+=noOfParticlesOnCPUs[p];
           }
     printf("Total number : %d",psum);

  }
  else // me!=0
  {
	  MPI_Send(&inum, 1,MPI_INT,0,10,world); // buf, count, datatype,
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////


  printf("Number of processors currently is %d\n________________________________________\n",nprocs);
  if(GENERATE_INTEGRATED_CONNECTIVITY_DIRS)
  {
	  if((currentTimeStep) == 1) // creates the directory for the current specific cpu only if the simulation has just started
	  {
	     //sprintf(str3,"exec rm -rf ./post/LiquidBridges"); // No need to clear post directory. LIGGGHTS does it automatically
	     sprintf(str3,"exec rm -rf ./post/processor*"); // No need to clear post directory. LIGGGHTS does it automatically
	     system(str3);
	     // system("mkdir ./post/LiquidBridges");
	     //sprintf(cpuDirectoryPathName,"post/LiquidBridges/processor%d",me);
	     sprintf(cpuDirectoryPathName,"post/processor%d",me);
	     sprintf(str1,"mkdir %s", cpuDirectoryPathName);
	     system(str1);

	    sprintf(str3,"exec rm -rf ./post/Connectivity");
	     system(str3);
	     system("mkdir ./post/Connectivity");
	     sprintf(cpuDirectoryPathName,"post/Connectivity/processor%d",me);
	     sprintf(str1,"mkdir %s", cpuDirectoryPathName);
	     system(str1);
	  //sprintf(filename,"post/Connectivity/processor%d/%g",me,(currentTimeStep)*(update->dt));
	   //system("echo hello>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
	  //printf(" Iteration number is %d \n ====================\n", update->ntimestep);
	  }

     //printf("I'm on CPU : %d . dump frequency is : %d , The ntimestep is %d\n",me, printFrequency, currentTimeStep);


  	  if ((currentTimeStep % printFrequency)==0)
  	  {
  		  printTimeStep=true;
  		  sprintf(filename,"post/processor%d/%g",me,(currentTimeStep)*(update->dt));
  		  // sprintf(filename,"%g",(currentTimeStep)*(update->dt));

  		  connectivityFile=fopen(filename,"w");
  		  fprintf(connectivityFile,"");
  		  fclose(connectivityFile);
  		  connectivityFile=fopen(filename,"a+");
  	  }
  }
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


  //fprintf(connectivityFile,"Shalom %d\n",me);

  vector <int> *adj;    // Pointer to an array containing adjacency lists

  if(me==0) adtSize=natoms; // set the size of the ADT accoding to the CPU. The head CPU will hold all particles data
   else adtSize=inum;

    try
    {

    	adj = new vector<int>[adtSize];
    }

    catch (std::bad_alloc& ba)
        {
          std::cerr << "Ziv watch out bad_alloc caught: " << ba.what() << '\n';
        }


  printNoOfParticlesOnCPU(me, adtSize); // print the particles distribution over the processors


  for(int t=0; t< inum; t++)
	  {
	  //if (me==0) adj[tag[t]].push_back(tag[t]); // to be changed - excluding first examined particle for head cpu - as the index serves as particle number
	            // on the root CPU no need to store for each particle his ID as its index plays this function.
	  if(me!=0)
	       adj[t].push_back(tag[t]); // Add real particles' IDs as first item for listed particles on current CPU vector
	       //printf("local no. is %d, Its global id is %d. which has %d connected liquid bridges\n", t,tag[t], int(adj[t].size()));
	  }

  //printf("I added vector of size %d. is it the size vector.size has - %d ??????\n", inum, int(adj[0].size()));

  int numberOfFoundBridges=0;
  if(printTimeStep) printf("Entering Time step : %g sec. on CPU %d \n",(currentTimeStep)*(update->dt),me);
//////////////////////////////////////////////////////////////////////////////////////////////// Ziv

  // loop over neighbors of my atoms
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    double xtmp = x[i][0];
    double ytmp = x[i][1];
    double ztmp = x[i][2];
    double radi = radius[i];
    int * jlist = firstneigh[i];
    int jnum = numneigh[i];
    touchpair = firsttouch[i];
    if (HISTFLAG) touch = firsttouch[i];
    double *allhist = firsthist[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;

      bool BRIDGE_EXIST=false; // a flag to mark whether a bridge was formed. ZG

      if (!(mask[i] & groupbit) && !(mask[j] & groupbit)) continue;

      double delx = xtmp - x[j][0];
      double dely = ytmp - x[j][1];
      double delz = ztmp - x[j][2];
      double radj = radius[j];
      double rsq = delx*delx + dely*dely + delz*delz;
      double r = sqrt(rsq);  //the distance between the centers of the particles

  //Calculate film thickness of individual particles

      double volPartFilm_i = liqOnParticle[i]+fourThirdPi*radi*radi*radi; //volume of particle + liquid film
      double volPartFilm_j = liqOnParticle[j]+fourThirdPi*radj*radj*radj; //volume of particle + liquid film

      double radsum = radi + radj;
      // if(radi != radj)
      //   error->one(FLERR,"Liquid transfer model only valid for equal-size particles - change transferratio accordingly!");

      double rstar_i =  pow( invFourThirdPi*volPartFilm_i, oneByThree); //Radius of particle+film
      double rstar_j =  pow( invFourThirdPi*volPartFilm_j, oneByThree); //Radius of particle+film
      double rfact   = (radi+radj) * (radi+radj);

      double filmh_i = rstar_i - radi;       //film thickness for i
      double filmh_j = rstar_j - radj;       //film thickness for i
      double filmh   = filmh_i + filmh_j;    //total film thickness

  //Calculation of separation distance between the particles
      double distsep = r - radi - radj;        //separation distance between the solid particle surfaces
      double hc      = *transferCoefficient;   // For model A, hc = Gamma number * dimensional shear rate

  //Calculate the effective radius, reference volume of particles and reference time (Model C)
      double reff    =  2.0 *radi *radj /radsum; //Effective radius of particles
      double tref    =  reff * liquidViscosity/surfaceTension; //Reference time calculated based on Gamma number
      double dp = 2* reff;
      //double vref    =  reff *reff *reff;        //Reference volume
      //double A       = a *(vref) /(sqrt(tref));  //Parameter used for Model - C3 (future work)

  //Calculation of surface area of particles (Surface coverage model_A - included in Model C2)
      double randSCR  =  random_equal->uniform() ;

  // To access the extra history value
      int allHistId = ((dnum + n_history_extra())*jj )+ dnum;

      int allHistId_wet = ((dnum + n_history_extra())*jj )+ dnum + 1;
      double *counter = &allhist[allHistId_wet];

      int allHistId_wet_i = ((dnum + n_history_extra())*jj )+ dnum + 2;
      double *counter_wet_i = &allhist[allHistId_wet_i];

      int allHistId_wet_j = ((dnum + n_history_extra())*jj )+ dnum + 3;
      double *counter_wet_j = &allhist[allHistId_wet_j];

//To Switch between models (A,B1,B2,C)
//******************** Model A **********************************//

      if(model_switch_flag == 0)
      {
        # include "fix_liquidtracking_instant_modelA.h"

    	  if(flux)
    	  {
    		  // Ziv . collecting data for Connectivity.
    		  // if there is a flux between the particles than there is a bridge ZG
    		  // this bridge is considered a bridge for connectivity mapping. No volume is transfered
    		  // from and to the bridge.

    		  BRIDGE_EXIST=true;
    		  numberOfFoundBridges++;
    		 printf("Model A : %d and %d share a liquid bridge\n", tag[i], tag[j]);
    		 if(BRIDGE_EXIST&& ((currentTimeStep % printFrequency)==0) )
    		 {
    			 if(GENERATE_INTEGRATED_CONNECTIVITY_DIRS)
    			 {
    				 fprintf(connectivityFile,"%d %d on processor %d which has a total of %d liquid bridges\n",tag[i],tag[j],me,numberOfFoundBridges);
    				 updateConnectivity(tag[i],tag[j], me, connectivityFile); // Ziv If there is a bridge than update the connectivity.
    			 }
    		   if (me==0) adj[tag[i]].push_back(tag[j]); // to be changed - excluding first examined particle for head cpu - as the index serves as particle number
    		   else
    		   	       adj[i].push_back(tag[j]);// Add real particles' IDs as first item for listed particles on current CPU vector
    		  }

    	  }

      }

//******************** Model B1 **********************************//

      if ( model_switch_flag ==1 ) //Christoph's Model
      {
        //Pull out the touch fix history
        double * touchfix        = &allhist[allHistId];
          liquid_exchange = false;
          //

        # include "fix_liquidtracking_instant_modelB1.h"

         //
        // Ziv  if ( !liquid_exchange ) printf("\n Entered B1 Model !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n");
        if ( !liquid_exchange ) continue;

        double mbridge_i = 0.5*liqOnParticle[i] * (1. - sqrt(1. - (radj*radj)/rfact) );
        double mbridge_j = 0.5*liqOnParticle[j] * (1. - sqrt(1. - (radi*radi)/rfact) );
        flux_i = ((mbridge_i+mbridge_j)*transfer_ratio - mbridge_i)*rmass[i]/dt;
        flux_j = ((mbridge_i+mbridge_j)*transfer_ratio - mbridge_j)*rmass[j]/dt;

     //   printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> flux_i = %g , flux_j = %g\n",flux_i,flux_j);
   //     printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> mbridge_i = %g , mbridge_j = %g\n",mbridge_i,mbridge_j);

        if((mbridge_i>0.0) || (mbridge_j>0.0))
        	{
        	 // Ziv . collecting data for Connectivity.
        	 // if there is a flux from either of the  particles than there is a bridge ZG
        	 // this bridge is considered a bridge for connectivity mapping.

        	BRIDGE_EXIST=true;
        	printf("Model B1 : %d and %d share a liquid bridge\n", tag[i], tag[j]);
        	numberOfFoundBridges++;
        	if(BRIDGE_EXIST&& ((currentTimeStep % printFrequency)==0) )
        	   {
        		if(GENERATE_INTEGRATED_CONNECTIVITY_DIRS)
        		{
        	    	     fprintf(connectivityFile,"%d %d on processor %d which has a total of %d liquid bridges\n",tag[i],tag[j],me,numberOfFoundBridges);
        	    	     updateConnectivity(tag[i],tag[j], me, connectivityFile); // Ziv If there is a bridge than update the connectivity.
        		}
        	    if (me==0) adj[tag[i]].push_back(tag[j]); // to be changed - excluding first examined particle for head cpu - as the index serves as particle number
        	    else
        	    	adj[i].push_back(tag[j]);// Add real particles' IDs as first item for listed particles on current CPU vector

        	    }
        	}


      }

//******************** Model B2 **********************************//

      if ( model_switch_flag == 2 ) //Bhageshvar's Instantaneous liquid bridge Model
      {
        double * bridgeVolume    = &allhist[allHistId];	//...the actual bridge volume
          liquid_exchange = false;

        //the liquid films are touching the first time - liquid drains instantaneously from the particles into the bridge
        # include "fix_liquidtracking_instant_modelB2.h"

        //Rupture calculation
        # include "fix_liquidtracking_rupturemodel.h"

        liqInBridge[i][0] += 0.5 * bridgeVolume[0];
        if (newton_pair || j < nlocal) liqInBridge[j][0] += 0.5 * bridgeVolume[0];
        printf("%d and %d share a liquid bridge\n", tag[i], tag[j]);


        if( (bridgeVolume[0]>0.0)|| (liqInBridge[j][0] >0.0) ||(liqInBridge[i][0]>0.0)) //	if(bridgeVolume[0]!=0.0)

        	{
        	  // Ziv . collecting data for Connectivity.
        	  // if there is bridge volume it will be mapped for connectivity.
        	BRIDGE_EXIST=true;
        	numberOfFoundBridges++;
        	}

        if(BRIDGE_EXIST&& ((currentTimeStep % printFrequency)==0) )
         {
             if(GENERATE_INTEGRATED_CONNECTIVITY_DIRS)
            	 {
            	 fprintf(connectivityFile,"%d %d on processor %d which has a total of %d liquid bridges\n",tag[i],tag[j],me,numberOfFoundBridges);
            	 updateConnectivity(tag[i],tag[j], me, connectivityFile); // Ziv If there is a bridge than update the connectivity.
            	 }

                if (me==0) adj[tag[i]].push_back(tag[j]); // to be changed - excluding first examined particle for head cpu - as the index serves as particle number
                else
                    adj[i].push_back(tag[j]);// Add real particles' IDs as first item for listed particles on current CPU vector
         }



        if( bridgeVolume[0]<0.0) printf("Error : Negative liquid bridge volume\n");  // Ziv

        if ( !liquid_exchange )continue;
      }

//******************** Model C **********************************//

      if ( model_switch_flag == 3 ) //Bhageshvar's rate based model for liquid bridge formation
      {
        //Pull out the bridge volume and counter
        double * bridgeVolume    = &allhist[allHistId]; //...the actual bridge volume
          liquid_exchange = false;
        double V_rough = MY_PI * (2 * reff) * ( 2 * reff) * roughnessLength * 2 * reff;

        if ( modelC_switch == 0 ) //Model-C1 Extended model with roughness effect
        {
          # include "fix_liquidtracking_instant_modelC1.h"
        }
        else if ( modelC_switch == 1 ) //Model-C2 Initial liquid bridge volume
        {
          # include "fix_liquidtracking_instant_modelC2.h"
        }
        else if ( modelC_switch == 2 ) //Model -C3 Surface coverage model
        {
          # include "fix_liquidtracking_instant_modelC3.h"         
        }
        else  //Model -C4 Future model for liquid bridge filling
        {
          # include "fix_liquidtracking_instant_modelC4.h"         
        }

        //Rupture calculation
          # include "fix_liquidtracking_rupturemodel.h"

        liqInBridge[i][0] += 0.5*bridgeVolume[0];
        if (newton_pair || j < nlocal) liqInBridge[j][0] += 0.5*bridgeVolume[0];

        if(bridgeVolume[0]!=0)
        {
        	//// Ziv . collecting data for Connectivity.
        	// if there is bridge volume it will be mapped for connectivity.
        	BRIDGE_EXIST=true;
        	numberOfFoundBridges++;
        	//adj[i].push_back(tag[j]); // Add real particles' IDs as first item for listed particles on current CPU vector

        if(BRIDGE_EXIST&& ((currentTimeStep % printFrequency)==0) )
          {
              //fprintf(connectivityFile,"%d %d on processor %d which has a total of %d liquid bridges\n",tag[i],tag[j],me,numberOfFoundBridges);
              //updateConnectivity(tag[i],tag[j], me, connectivityFile); // Ziv If there is a bridge than update the connectivity.
        	if (me==0) adj[tag[i]].push_back(tag[j]); // to be changed - excluding first examined particle for head cpu - as the index serves as particle number
        	else
        	    adj[i].push_back(tag[j]);// Add real particles' IDs as first item for listed particles on current CPU vector
          }

        }// end of bridgeVoume calculation

        if(!liquid_exchange)  continue;

      }//end of Model - C

//******************** End of Model C **********************************//
// Conductive flux calculation for models B1,B2 and C
      if(model_switch_flag!=0)
      {
        if ( !cpl_flag )
        {
          liqFlux[i] += flux_i;
          if (newton_pair || j < nlocal) liqFlux[j] += flux_j;
          condFlux_i[0] = flux_i*delx;
          condFlux_i[1] = flux_i*dely;
          condFlux_i[2] = flux_i*delz;

          condFlux_j[0] = flux_j*delx;
          condFlux_j[1] = flux_j*dely;
          condFlux_j[2] = flux_j*delz;

          condLiqFlux[i][0] += 1.0*condFlux_i[0];
          condLiqFlux[i][1] += 1.0*condFlux_i[1];
          condLiqFlux[i][2] += 1.0*condFlux_i[2];
          if (newton_pair || j < nlocal) condLiqFlux[j][0] += 1.0*condFlux_j[0];
          if (newton_pair || j < nlocal) condLiqFlux[j][1] += 1.0*condFlux_j[1];
          if (newton_pair || j < nlocal) condLiqFlux[j][2] += 1.0*condFlux_j[2];
        }
        if(modelC_switch !=2)
        { 
          BridgeNo[i][0] += 0.5*n_bridge;
            if (newton_pair || j < nlocal) BridgeNo[j][0] += 0.5*n_bridge; 
        }
        if(modelC_switch ==2)
        { 
          surfaceCoverage[i][0] += n_wet/2.0;
            if (newton_pair || j < nlocal) surfaceCoverage[j][0] += 0.5*n_wet; 
        }
          if(cpl_flag && cpl) cpl->add_heat(i,j,flux);
      } // end of calculation of conductive flux calculation

      /*if(BRIDGE_EXIST&& ((currentTimeStep % printFrequency)==0) )
    	  {
    	     fprintf(connectivityFile,"%d %d on processor %d which has a total of %d liquid bridges\n",tag[i],tag[j],me,numberOfFoundBridges);
    	     updateConnectivity(tag[i],tag[j], me, connectivityFile); // Ziv If there is a bridge than update the connectivity.
    	  }*/
    } // end of j - loop

    // sort the adj lines to allow later union of all
    if(me==0)
    {
    	std::sort (adj[tag[i]].begin(), adj[tag[i]].end()); // on root processor the index is the particle's I.D. thus particles are sorted from first location
    }
    else //me!=0
    {
    	 std::sort (adj[i].begin()+1, adj[i].end()); // on each of the non root processors the particle's I.D. is the first element thus srting is done beyond it.
    }

  } //end of i - loop

//
// printing the current liquid bridges collected data

  //MPI_Barrier(world);

  vector<int>::iterator connectPath;
  int linkedParticle;
  std::stringstream ss;

  	  ss.str("");
  	  ss.clear();

  if((currentTimeStep % printFrequency)==0)
  {
	  for(int t=0;t<inum;t++)
      {
		  for(connectPath = adj[t].begin(); connectPath != adj[t].end(); ++connectPath)
		  {
  	        linkedParticle=*connectPath;
  	        //printf("%d ",linkedParticle);
  	        //fprintf(connectivityFile,"%d ",linkedParticle);
  	        ss << linkedParticle << " ";
  	      //MPI_Barrier(world);
		  }
		  //printf("\n");
		  ss<<endl;
     //printf("The file pointer is %d  .\n",connectivityFile);
     //fprintf(connectivityFile,"%s","\n");
     //printf(" vector.size has - %d ??????\n", int(adj[t].size()));
      }
	  cout<<"Printing connectivity ss"<<endl;
	  printf("-------------------------------------------------------------------\n");
	  cout<<ss.str();
	  cout<<"End of printing for CPU "<<me<<endl;

	  if(GENERATE_INTEGRATED_CONNECTIVITY_DIRS) // printing to a file only when required to allow a
	  {
		  sprintf(connectivityFilename,"post/Connectivity/processor%d/%g",me,(currentTimeStep)*(update->dt));

		  ofstream connectivityFileStream(connectivityFilename, ios::out );
	    	  	   	  // exit program if ifstream could not open file
	    	  	   if ( !connectivityFileStream )
	    	  	   	{
	    	  	   	    cerr << connectivityFilename <<" File could not be opened" << endl;
	    	  	   	   			//	exit( 1 );
	    	  	   	} // end if


	    	  	   connectivityFileStream<<ss.str();
	    	  	   connectivityFileStream.close();
	  }

  }



  //fprintf(connectivityFile," X ");
  if(newton_pair) fix_liqFlux->do_reverse_comm();
  if(newton_pair) fix_condLiqFlux->do_reverse_comm();
  if(newton_pair) fix_liqInBridge->do_reverse_comm();
  if(newton_pair) fix_BridgeNo->do_reverse_comm();
  if(newton_pair) fix_surfaceCoverage->do_reverse_comm();
//  if(newton_pair) fix_BridgeTime->do_reverse_comm();

  if((currentTimeStep % printFrequency)==0)
  {
	  printf("End of iteration of time step : %g on cpu : %d. number of liquid bridges on cpu is %d\n",(update->ntimestep)*(update->dt) ,me,numberOfFoundBridges );

	  if(GENERATE_INTEGRATED_CONNECTIVITY_DIRS)
		  {
				  fprintf(connectivityFile,"%d",numberOfFoundBridges); //   prints liquid bridges on CPU for each timestep
				  fclose(connectivityFile);
		  }

  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Moving data from CPUs to root CPU



    //vector <int> passingVector;

    int passingVectorSize;
    //MPI_Status status; //             already declared
    int incomingVectorSize;

    if (me == 0) passingVectorSize= MAX_LIQUID_BRIDGES;
    else passingVectorSize =MAX_LIQUID_BRIDGES;

    //passingVector = new vector <int> [passingVectorSize];

    //for(int y=0;y<passingVectorSize;y++ ) passingVector.push_back(y+10*me);// filling every vector on each CPU with unique id

   // if(me==0) for(int y=0;y<passingVectorSize;y++ ) recieved_vector.push_back(me);

    //printf("Printing all created vectors on CPU %d ", me);
    //for(int y=0; y<passingVector.size();y++) cout << passingVector[y]<<" "; cout<<endl;

    //for(int y=0;y<passingVectorSize;y++ ) passingVector.push_back(me);

    if (me==0)
     {
  	        ////   MPI_Recv(&incomingVectorSize, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, world, &status);
  	        //// cout <<"I'm the root. I got value of : "<<incomingVectorSize<<", MPI_ANY_SOURCE = "<<MPI_ANY_SOURCE<<". MPI_ANY_TAG is :" << MPI_ANY_TAG<<endl;
  	  	  //MPI_Barrier(world);
  	  	  // vector <int> *received_vector;
  	  //received_vector = new vector<int>[passingVectorSize];
  	  	  //vector <int> received_vector;
  	  int bridges[MAX_LIQUID_BRIDGES];
  	  vector <int> received_vector(MAX_LIQUID_BRIDGES);
  	  //received_vector = new vector<int>[10];
  	  int MAXIMUM_Merged_Vectors=50;
  	  vector <int> numbers;

  	std::vector<int> v(MAXIMUM_Merged_Vectors);

  	  	  	  	  	 numbers.push_back(12222); // for testing only
  	                 numbers.push_back(7); // for testing only
  	                 numbers.push_back(24); // for testing only
  	                 numbers.push_back(3); // for testing only
  	                 numbers.push_back(6); // for testing only
  	               std::sort (numbers.begin(), numbers.end()); //sorting the sent vector to allow later union with other particles already registered liquid bridges
  	               printf("Numbers on vector numbers are : ");
  	               for(int y=0; y<numbers.size();y++) printf("%d     ",numbers[y]); cout<<endl;

  	  for(int p=1; p<nprocs;p++)
  	  {

  		std::vector<int>::iterator it;
  		  printf("Receiving from cpu : %d\n",p);
  		  MPI_Recv(&received_vector[0], MAX_LIQUID_BRIDGES, MPI_INT, p, MPI_ANY_TAG, world, &status);

  		  received_vector[3]=12222;received_vector[4]=3;// plant a duplicate

  		  //\\\ MPI_Recv(&bridges[0], 10, MPI_INT, 2, MPI_ANY_TAG, world, &status);
  	  	  //   MPI_Recv(&received_vector[0], 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, world, &status);
  	  	   cout<<"<<<<< ***** >>>>> I'm the root - Printing values of received vector ---> " ;
  	  	   // for(int y=0; y<passingVectorSize;y++) printf("%d     ",bridges[y]); cout<<endl;
             for(int y=0; y<MAX_LIQUID_BRIDGES;y++) printf("%d     ",received_vector[y]); cout<<endl;
             std::vector<int>::iterator chk_it;
             //chk_it=find(received_vector.begin(), received_vector.end(), 45);

             if( find(received_vector.begin(), received_vector.end(), 45)== received_vector.end()) printf("did not Found 45!!!!!!!!!!!! on CPU %d. received_vector size is :%d\n",p,(int)received_vector.size());
          		   else printf("Found 45!!!!!!!!!!!! on CPU %d. received_vector size is :%d\n",p,(int)received_vector.size());

             it=std::set_union (received_vector.begin(), received_vector.end(), numbers.begin(), numbers.end(), v.begin());

               v.resize(it-v.begin());
               printf("\n The unified vector on cpu %d has %d elements : \n", me, v.size());

               for (it=v.begin(); it!=v.end(); ++it)
                   std::cout << ' ' << *it;
               received_vector=v; // repeat testing making sure same print
               printf("\n after assignement all v to received vector we get on cpu %d : \n", me);
               for (it=received_vector.begin(); it!=received_vector.end(); ++it)
                   std::cout << ' ' << *it;
               cout<<endl;
               v.clear();
               received_vector.clear();
               v.resize(MAXIMUM_Merged_Vectors);
               received_vector.resize(MAX_LIQUID_BRIDGES);


  	  }	  	    // delete [] received_vector;

  	  	  // for(int y=0; y< mediator_vector.size();y++) passingVector.push_back(mediator_vector[y]); //putting the received values to the me==0 CPU

  	  	//   for(int y=0;y< passingVector.size();y++) cout << passingVector[y]<< " "; // print updated vector
  	  	 //  cout<<endl;
  	  //delete [] received_vector;

     }
     else // me!=0
     {
  	   //vector <int> passingVector;
  	   int passingVector[MAX_LIQUID_BRIDGES];
  	   vector <int> sent_vector(MAX_LIQUID_BRIDGES);

  	  // MPI_Request reqs[2];

  	   	 //// MPI_Send(&passingVectorSize,1,MPI_INT,0,10,world);
  	   	  //////////////////////printf("I'm cpu %d, sending the value %d\n", me, passingVectorSize);
  	  //  for(int y=0;y<passingVectorSize;y++ ) passingVector[y]=10*me+y;//passingVector.push_back(me);// filling every vector on each CPU with unique id
  	   	   for(int y=0;y<passingVectorSize;y++ ) sent_vector[y]= -( 10*me+y); //sent_vector.push_back(10*me+y);

  	   	 printf("I'm sending informatiom from me - %d to root \n", me);

  	   	//for(int y=0; y<passingVectorSize;y++) printf("%d ",passingVector[y]); cout<<endl;
  	   	for(int y=0; y<passingVectorSize;y++) printf("%d \n",sent_vector[y]);
  	    std::sort (sent_vector.begin()+1, sent_vector.end()); //sorting the sent vector to allow later union with other particles already registered liquid bridges
  	    printf("and the sorted values on cpu %d are :\n",me);
  		for(int y=0; y<passingVectorSize;y++) printf("%d ",sent_vector[y]); printf("\n");
  	   	//MPI_Barrier(world);
  	   	 // MPI_Isend(&passingVector[0], 10,MPI_INT,0,10,world,&reqs[1]); // buf, count, datatype,
  	   //	MPI_Isend(&value_sent, 1, MPI_INT, send, tag, MPI_COMM_WORLD, &reqs[1]);
  	   	    // MPI_Send(&passingVector[0], 10,MPI_INT,0,10,world); // buf, count, datatype,
  	   	  MPI_Send(&sent_vector[0], MAX_LIQUID_BRIDGES,MPI_INT,0,10,world); // buf, count, datatype,

     }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  delete [] adj;
  //delete [] passingVector;
}

/* --------------------------------------------------------------------- */
// end_of_step function currently switched off to be used if model C3 is used
//  #include "fix_liquidtracking_instant_modelC3_endofstep.h"
/* ----------------------------------------------------------------------
   register and unregister callback to compute
------------------------------------------------------------------------- */

void FixLiquidTrackingInstant::register_compute_pair_local(ComputePairGranLocal *ptr)
{

   if(cpl != NULL)
      error->all(FLERR,"Fix liquidTracking/instant allows only one compute of type pair/local");
   cpl = ptr;
}

void FixLiquidTrackingInstant::unregister_compute_pair_local(ComputePairGranLocal *ptr)
{

   if(cpl != ptr)
       error->all(FLERR,"Illegal situation in FixLiquidTrackingInstant::unregister_compute_pair_local");
   cpl = NULL;
}

void updateConnectivity(int & i_globalID, int & j_globalID,int & currentProcessor, FILE *connectivityFile)
{
printf("particle i = %d, particle j = %d are connected on processor %d \n", i_globalID, j_globalID,currentProcessor);
//fprintf(connectivityFile,"%d %d on processor %d\n",i_globalID,j_globalID,currentProcessor);

//if( bridgeVolume[0] > 0 ) fprintf(connectivityFile,"%d ",jglobal);
}

////////////////////////////////////////////////////////////////////////////////////////////////

void printNoOfParticlesOnCPU(int currentCPU, int adtSize) { printf("CPU %d, adtSize = %d\n",currentCPU, adtSize);}

///////////////////////////////////////////////////////////////////////////////////////////////////

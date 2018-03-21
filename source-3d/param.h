namespace MODEL
{
  const double C0 = 0.0;
  const int N_GRIDS_1D = 210;
  const int N_GRIDS_3D = N_GRIDS_1D*N_GRIDS_1D*N_GRIDS_1D;
  const int N_VARIABLES = 4+N_GRIDS_1D*N_GRIDS_1D*N_GRIDS_1D*6;
  const double X_MAX = 105;
  const double X_MIN = -105;
  // Parameters to control Integrate ODS precision
  const double dt = 0.05;
  const double t_max = 102;
  const size_t n_print = 20;
  // Parameters to control the absorption boundary condition
  const double X_CUTOFF = 28;
  const double X_MIDDLE = 16;
  // Initialize OMEGA
  const double OMEGA = 0.25;
  // Initialize strength of P
  const double P_MAX =  0.025; //0.025
  const double SIGMA = 0.50; //0.3
  // FGR rate in 3D
  const double GAMMA = P_MAX*P_MAX*OMEGA*OMEGA*OMEGA/3/3.1415926;
  
  // Parameters to control the place to store varibales
  // Some starting points;
  const size_t C1_START = 0;
  const size_t C2_START = 2;
  const size_t Ex_START = 4;
  const size_t Ey_START = Ex_START + N_GRIDS_3D;
  const size_t Ez_START = Ey_START + N_GRIDS_3D;
  const size_t Bx_START = Ez_START + N_GRIDS_3D;
  const size_t By_START = Bx_START + N_GRIDS_3D;
  const size_t Bz_START = By_START + N_GRIDS_3D;

  /*const double start_time[5] = {0, 1500*dt, 2000*dt, 2500*dt, 3000*dt};*/
  /*const double end_time[5] = {1500*dt, 2000*dt, 2500*dt, 3000*dt, 3500*dt};*/
  const double start_time[5] = {0, 20.0, 40.0, 60.0, 80.0};
  const double end_time[5] = {20.0, 40.0, 60.0, 80.0, 100.0};

}


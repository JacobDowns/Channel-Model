#### Physical Constants ###     
     
# Seconds per day
spd = 60.0 * 60.0 * 24.0
# Seconds in a year
spy = spd * 365.0                            
# Density of water (kg / m^3)
rho_w = 1000.0  
# Density of ice (kg / m^3)
rho_i = 910.0
# Gravitational acceleration (m / s^2)
g = 9.81 
# Flow rate factor of ice (1 / Pa^3 * s) 
A = 5.0e-25
# Average bump height (m)
h_r = 0.1
# Typical spacing between bumps (m)
l_r = 2.0
# Sheet width under channel (m)
l_c = 2.0          
# Sheet conductivity (m^(7/4) / kg^(1/2))
k = 1e-2
# Channel conductivity (m^(7/4) / kg^(1/2))
k_c = 1e-1
# Pressure melting coefficient (J / (kg * K))
c_t = 7.5e-8
# Latent heat (J / kg)
L = 3.34e5
# Exponents 
alpha = 5. / 4.
beta = 3. / 2.
delta = beta - 2.0
# Regularization parameters for convergence
phi_reg = 1e-15
N_reg = 1e-15
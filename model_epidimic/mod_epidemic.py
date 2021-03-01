# author Dario Latorre
# Prueba IETS
# 28 02 2021
import numpy as np
import pandas as pd

class MEpidemic():  
  def __init__(self):
    """
    Create a instance of model epidemic:    
    After of create a model epidemic  instance, it's should ould load parameters with the method load parameters.
    """
    self.parameters = None
    self.cond_init = None
    self.cond_bound = None
    self.array_t = None
    self.size_grid_t = None
    self.array_age = None
    self.dataframe_solutions = None
    self.time_for_predict = None
  def load_parameters_conditions(self, dic_parameters, cond_init, cond_bound, size_grid_t, time):
    """
    Load the parameters need for model and the initial and boundary conditions for model.

    dic_parameters: A dictionary with parameters for model. View information for the respective model.

    cond_init:  A dictionary with initial conditions  for model. View information for the respective model.

    cond_bound: A list with boundary conditions for model. View information for the respective model.

    size_grid_t: a number between 2 and 9999 for the solution numerical.

    """
    self.parameters = dic_parameters
    self.cond_init = cond_init
    self.cond_bound = cond_bound
    self.size_grid_t = size_grid_t
    self.time_for_predict = time

    self.array_t = np.linspace(0,1,size_grid_t)
    self.array_age = np.arange(0,(dic_parameters['age_max'] +1),1)
    list_names = [f'age_{age}_state_{state}' for age in self.array_age for state in self.LIST_STATES]
    array_data = np.zeros(len(list_names)).reshape(1,-1)
    self.dataframe_solutions =pd.DataFrame(data = array_data, columns= list_names)
    
  def print_model(self):
    """
    Print  the graph model.
    """
    pass
  
class MEpidemicPertussisNeth2001(MEpidemic):
  LIST_PAR_MODEL = ['v', 'sigma_v', 'sigma_i', 'rho_1', 'rho_2', 'lamb_f_1', 'lamb_f_2','mu_s_1', 'mu_s_2', 'mu_i_1', 'mu_i_2', 'mu_v', 'mu_r']
  LIST_STATES = ['S1','V','I1','S2','I2','R']
  """
  Create a instance of a model epidemic based on the paper 
  "A model based evaluation of the 1996–7 pertussis epidemic in the Netherlands"
   by M van Boven 1, H E de Melker, J F Schellekens, M Kretzschmar in 2001. 

  Although the model is based in the paper considerations for the parameters are diferents, 
  in this instance some parameters has a free greather that  in the paper and the method of solution
  can have a diferent approach.
  

  The parameters for model are:
  1.  age_max:   Max age for analize.
  2.  sigma_v:   Iterable object of length (age_max + 1) with Rate of loss of immunity after vaccination for each each age. 
  3.  sigma_i:   Iterable object of length (age_max + 1) with Rate of loss of immunity after infection.
  4.  rho_1:     Iterable object of length (age_max + 1) with Rate of loss of primary infection.
  5.  rho_2:     Iterable object of length (age_max + 1) with Rate of loss of secondary infection.
  6.  v:         Iterable object of length (age_max + 1) with Rate of vaccinated.
  7.  mu_s_1:    Iterable object of length (age_max + 1) with Rate of mortality for susceptibles not vaccinated.
  8.  mu_s_2:    Iterable object of length (age_max + 1) with Rate of mortality for susceptible individuals whose immune system has been primed before.
  9.  mu_i_1:    Iterable object of length (age_max + 1) with Rate of mortality for infected  individuals with primary infection.
  10. mu_i_2:    Iterable object of length (age_max + 1) with Rate of mortality for infected  individuals with secondary infection.
  11. mu_v:      Iterable object of length (age_max + 1) with Rate of mortality for vaccinated individuals.
  12. mu_r:      Iterable object of length (age_max + 1) with Rate of mortality for  protected indivduals after natural infection.
  13. lamb_f_1:  Iterable object of length (age_max + 1) with Rate of force  in suscpetible group 1.
  14. lamb_f_2:  Iterable object of length (age_max + 1) with Rate of force  in suscpetible group 2.


  The initial conditions for model is the conditions in t=0 for the set of states of model, in this case 6. Then:

  cond_init: Is a dictionary with 6 keys values with names the states  S1,S2,I1,I2,V,R. In the dictionary each key have a list of size (age_max +1)
             that indicate the values initial for each state and age.

  cond_bound: Is a list with conditions for S1(0,time_years) for time_years > 0  (The value in 0 is having into acount in the cond_init). 
              Each value in the list correspond to borns that in S1.
  """
  
  def solution_equations_1_year(self, age, cond_init):
    """ 
    Find  solution for system equations for a fix age and one period of 1 year. 
    The number of subdivisions of period is size_grid_t.

    return A np.array of shape size_grid_t X number of states, in this case 6.
    """
    dic_params = {par : self.parameters[par] for par in self.LIST_PAR_MODEL}
    solution = np.zeros((self.size_grid_t,6))
    solution[0,:] = cond_init
    dt_ap = 1/self.size_grid_t
    for t in range(1,self.size_grid_t):
      S1 = solution[t-1,0]
      V = solution[t-1,1]
      I1 = solution[t-1,2]
      S2 = solution[t-1,3]
      I2 = solution[t-1,4]
      R = solution[t-1,5]
      equations = {0:-dic_params['v'][age]*S1-dic_params['lamb_f_1'][age]*S1 - dic_params['mu_s_1'][age]*S1,
                   1:dic_params['v'][age]*S1 - dic_params['sigma_v'][age]*V - dic_params['mu_v'][age]*V,
                   2:dic_params['lamb_f_1'][age]*S1 - dic_params['rho_1'][age]*I1 - dic_params['mu_i_1'][age]*I1,
                   3:dic_params['sigma_v'][age]*V + dic_params['sigma_i'][age]*R -dic_params['lamb_f_2'][age]*S2 -  dic_params['mu_s_2'][age]*S2,
                   4:dic_params['lamb_f_2'][age]*S2 - dic_params['rho_2'][age]*I2 - dic_params['mu_i_2'][age]*I2,
                   5:dic_params['rho_1'][age]*I1 + dic_params['rho_2'][age]*I2 -dic_params['sigma_i'][age]*R - dic_params['mu_r'][age]*R
                   }
      for eq in range(6):
         solution[t,eq] = max(solution[t-1,eq] + equations[eq]*dt_ap,0.0001)
    return solution

  def fill_zeros_df(self):
    n_rows = (self.time_for_predict)*(self.size_grid_t)
    index_df = [f'{time:03d}_time_grid_{grid:07d}' for grid in range(self.size_grid_t) for time in range(self.time_for_predict)]
    columns = self.dataframe_solutions.columns
    n_col = len(columns)
    array_data = np.zeros((n_rows, n_col))
    return pd.DataFrame(data = array_data, columns= columns, index = index_df)


  def run_model(self):
    df = self.fill_zeros_df()
    for t in range(self.time_for_predict):
      index_fill = [ind for ind in df.index if f'{t:03d}_time' in ind]
      mask = [ind in index_fill for ind in df.index]
      for age in self.array_age:
        list_col_names = [col for col in df.columns if f'age_{age}_' in col]
        if t == 0:
          conditions_initials = [self.cond_init[state_init][age] for state_init in self.LIST_STATES]
        else:
          if age ==0:
            S_1_init = self.cond_bound[t]
            condition_initials = np.array([S_1_init,0,0,0,0,0])
          else:
            mask_index_initial = (df.index.values == f'{(t-1):03d}_time_grid_{(self.size_grid_t-1):07d}')
            list_col_names_pre =  [col for col in df.columns if f'age_{age-1}_' in col]
            conditions_initials = df.loc[mask_index_initial,list_col_names_pre].values

        solutions = self.solution_equations_1_year(age,conditions_initials)
        df.loc[mask,list_col_names] = solutions

    df.sort_index(inplace=True)
    self.dataframe_solutions = df
    return df

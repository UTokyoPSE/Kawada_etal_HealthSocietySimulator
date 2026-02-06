"""
Source code attached for the manuscript:

"Development of a health care system simulator for Japan toward efficient allocation of medical resources" by
Yusuke Kawada, Yusuke Hayashi, and Hirokazu Sugiyama

This code corresponds to the compartment model representing each disease progression.
Details of the model are described in "Methods / Country adaptation module / 2. Analyze disease trends".
The output figure corresponds to "Supplementary Information S2-S35.

Created by Yusuke Kawada/Sugiyama-Badr Lab/The University of Tokyo
Last saved on Feb 04 2026
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import re

group_num = 15
group_width = 5
t_spans = [(2001, 2004), (2004, 2007), (2007, 2010), (2010, 2013), (2013, 2016), (2016, 2019)]

input_dir = os.path.join('input', 'For_each_disease_progression')
output_base_dir = os.path.join('output', 'For_each_disease_progression')
os.makedirs(output_base_dir, exist_ok=True)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
font_size = 18

def each_disease_calc(disease_name):
    def state_change(t, X, beta_all, mu_all, beta, mu):
        population_now, healthy_now, care_now, care_disease_now = np.split(X, [group_num, 2*group_num, 3*group_num])
        # get aging_flow
        now_row = int(start_row + 1 + np.floor(t - start_year)) # row having # of birth in the year t
        aging_flow = np.zeros(group_num + 1)

        # aging_flow = [v_0, v_1, ....., v_n]
        for i in range(group_num + 1):
            aging_flow[i] = df_birth[now_row - group_width * i][1]

        # add x_H,0 x_C,0 to healthy_now, care_now
        healthy_now = np.insert(healthy_now, 0, healthy_0)
        care_now = np.insert(care_now, 0, care_0)
        care_disease_now = np.insert(care_disease_now, 0, care_disease_0)
        alive_not_disease = healthy_now + care_now - care_disease_now

        # model equation
        d_population = np.zeros(group_num)
        d_healty = np.zeros(group_num+1)
        d_care = np.zeros(group_num+1)
        d_care_disease = np.zeros(group_num+1)
        d_death_from_care = np.zeros(group_num+1)
        d_death_by_disease = np.zeros(group_num+1)
        d_death_by_others = np.zeros(group_num+1)
        for i in range(group_num):
            d_population[i] = aging_flow[i] - aging_flow[i+1]

        for i in range(1, group_num+1):
            d_healty[i] = (healthy_now[i-1]-healthy_now[i]) * aging_flow[i-1] / population_now[i-1] - beta_all[i] * healthy_now[i] # population[i-1] = N_i
            d_care[i] = (care_now[i-1]-care_now[i]) * aging_flow[i-1] / population_now[i-1] - mu_all[i] * care_now[i] + beta_all[i] * healthy_now[i]

            # for death division
            d_death_from_care[i] = mu_all[i] * care_now[i] # overall death from care
            d_death_by_disease[i] = mu[i] * care_disease_now[i] # death by the disease
            d_death_by_others[i] = (d_death_from_care[i] - d_death_by_disease[i]) # death by others

            # change in people with the disease
            d_care_disease[i] = (care_disease_now[i-1]-care_disease_now[i]) * aging_flow[i-1] / population_now[i-1] # population change term
            d_care_disease[i] += beta[i] * alive_not_disease[i] # new cases
            d_care_disease[i] += -mu[i] * care_disease_now[i] - d_death_by_others[i] * care_disease_now[i] / care_now[i] # death

        # change healthy, care, death
        d_healty = np.delete(d_healty, 0)
        d_care = np.delete(d_care, 0)
        d_care_disease = np.delete(d_care_disease, 0)
        
        dX = np.append(d_population, d_healty)
        dX = np.append(dX, d_care)
        dX = np.append(dX, d_care_disease)

        return dX

    def k_make(ks):
        # define beta_all, mu_all here !Note that beta_all0, mu_all0 is included in beta_all, mu_all
        k1, k2, k3, k4, k5, k6= ks
        beta_all = np.zeros(group_num+1)
        mu_all = np.zeros(group_num+1)
        for i in range(5):
            beta_all[i] = k1
            mu_all[i] = k4
        for i in range(5, 12):
            beta_all[i] = k2
            mu_all[i] = k5
        for i in range(12, group_num + 1):
            beta_all[i] = k3
            mu_all[i] = k6
        
        return beta_all, mu_all
    
    output_dir = os.path.join(output_base_dir, disease_name)
    os.makedirs(output_dir, exist_ok=True)
    for i_t in range(len(t_spans)):
        t_span = t_spans[i_t]
        start_year = t_span[0]
        end_year = t_span[1]

        # Specify the file path
        file_path_birth = os.path.join(input_dir, '00_basic', 'birth_num.csv') #file path of birth number data
        file_path_start = os.path.join(input_dir, '99_All', f'X_all_{start_year}.csv') #file path of input data
        file_path_start_disease = os.path.join(input_dir, disease_name, f'X_{disease_name[:2]}_{start_year}.csv') #file path of input data(for specific disease)
        file_path_end_disease = os.path.join(input_dir, disease_name, f'X_{disease_name[:2]}_{end_year}.csv') #file path of measured data(for specific disease)
        file_path_deathnum = os.path.join(input_dir, "99_Death", f"death_{start_year}.csv")
        file_path_fitresult_all = os.path.join(input_dir, '00_transition_rates', 'fitted_transition_rates_all.csv') #file path of fitted transition rate for overall health
        file_path_fitresult_mu = os.path.join(input_dir, '00_transition_rates', f'mu_fitting_results_{start_year}.csv') #file path of fitted beta
        file_path_fitresult_beta = os.path.join(input_dir, '00_transition_rates', f'beta_fitting_results_{start_year}.csv') #file path of fitted mu
        file_path_ymax = os.path.join(input_dir, '00_basic', 'y_max_suggested.csv') # file path of suggested ymax for each disease

        # Read the CSV file
        # df_birth->0:year, 1:birth_num
        # df_start, df_end->0:start_age, 1:end_age, 2:healthy, 3:care, 4:death
        df_birth = pd.read_csv(file_path_birth)
        df_start = pd.read_csv(file_path_start)
        df_start_disease = pd.read_csv(file_path_start_disease)
        df_end_disease = pd.read_csv(file_path_end_disease)
        df_deathnum = pd.read_csv(file_path_deathnum, index_col=0)
        df_fitresult_all = pd.read_csv(file_path_fitresult_all, index_col=0)
        df_fitresult_mu = pd.read_csv(file_path_fitresult_mu, index_col=0)
        df_fitresult_beta = pd.read_csv(file_path_fitresult_beta, index_col=0)
        df_ymax = pd.read_csv(file_path_ymax, index_col=0)
        df_birth = df_birth.values
        
        # read fitting result
        fitresult_all = df_fitresult_all.loc[start_year].values[1:] # overall
        fitresult_mu = df_fitresult_mu.loc[disease_name].values # beta for target disease
        fitresult_beta = df_fitresult_beta.loc[disease_name].values # mu for target disease
        ymax_suggested = df_ymax.loc[disease_name].values[0]

        # covert data to １D array
        healthy_start = df_start.iloc[0:group_num, [2]].values.flatten()
        care_start = df_start.iloc[0:group_num, [3]].values.flatten()
        care_disease_start = df_start_disease.iloc[0:group_num, [3]].values.flatten()
        care_disease_end = df_end_disease.iloc[0:group_num, [3]].values.flatten()

        # made asuumption that x_healthy, x_care at age 0 is same as the start year
        healthy_0 = healthy_start[0]
        care_0 = care_start[0]
        care_disease_0 = care_disease_start[0]

        # make fist population
        population_start = np.zeros(group_num)
        start_row = 0 # row having # of birth in start_year
        while df_birth[start_row][0] < start_year:
            start_row += 1

        for i in range(group_num):
            for j in range(group_width):
                population_start[i] += df_birth[start_row - group_width * i - j][1]
        
        beta_all, mu_all = k_make(fitresult_all)
        beta, mu = k_make(np.append(fitresult_beta, fitresult_mu))

        # For mu calculation-----------------------------------------------------------------------------------------------------------------------
        # divide death number into the target disease and others
        death_num_disease = df_deathnum.loc[re.sub(r"\d+_", "", disease_name)].values
        death_num_other = df_deathnum.loc["SUM"].values - death_num_disease

        d_death_from_care = np.zeros(group_num)
        death_num_disease_now = np.zeros(group_num)
        death_num_other_now = np.zeros(group_num)
        death_by_disease = np.zeros(group_num)

        # calculate v_i, D
        for i in range(group_num):
            d_death_from_care[i] = mu_all[i+1] * care_start[i] * population_start[i]
            death_num_disease_now[i] = death_num_disease[i] * d_death_from_care[i] / (death_num_disease[i] + death_num_other[i]) / population_start[i]
            death_num_other_now[i] = death_num_other[i] * d_death_from_care[i] / (death_num_disease[i] + death_num_other[i]) / population_start[i]
            death_by_disease[i] = death_num_disease_now[i] * population_start[i]
        
        death_calculated = np.zeros(group_num)
        for i in range(group_num):
            death_calculated[i] = mu[i+1] * care_disease_start[i] * population_start[i]

        # make graph
        X = []
        for i in range(group_num):
            X.append(str(i*5) + '\u2013' + str(i*5+4))
        X = np.array(X)
        plt.figure(figsize=(6, 4.5))
        plt.plot(X, death_by_disease, color='#000000', marker='.', ls='', label='diseaese_measured')
        plt.plot(X, death_calculated, color='#000000', ls='-', label='disease_simulated')
        plt.xlabel('Age', fontsize=font_size)
        plt.ylabel('# of death [person yr$^{-1}$]', fontsize=font_size)
        plt.xticks(np.arange(len(X)), rotation=90)
        plt.tick_params(labelsize=font_size)
        plt.title(f'{start_year}\u2013{end_year}', fontsize=font_size)
        plt.savefig(os.path.join(output_dir, f'v_D_{disease_name[:2]}_calc_{start_year}-{end_year}.tif'), bbox_inches='tight', dpi=300)
        plt.show()

        # output fitting result to csv
        df_calc = df_end_disease.copy()
        df_calc.iloc[0:group_num, [2]] = death_calculated
        df_calc.drop(columns=df_calc.columns[3:], inplace=True)
        df_calc.drop(index=df_calc.index[group_num:], inplace=True)
        df_calc.to_csv(os.path.join(output_dir, f'v_D_{disease_name[:2]}_calc_{end_year}_from_{start_year}.csv'), index=False)

        # for beta calculation------------------------------------------------------------------------------------------------------------------------
        # make population, x_healthy, x_care into one array
        # [N_1, N_2, ....., N_n, x_h1, x_h2, ....., x_hn, x_c1, x_c2, ....., x_cn]
        init = np.append(population_start, healthy_start)
        init = np.append(init, care_start)
        init = np.append(init, care_disease_start)
        sol = solve_ivp(state_change, t_span, init, args=(beta_all, mu_all, beta, mu), t_eval=[end_year], method='LSODA')

        population_calc, healthy_calc, care_calc, care_disease_calc = np.split(sol.y.squeeze(), [group_num, 2*group_num, 3*group_num])
        alive_not_disease_calc = healthy_calc + care_calc - care_disease_calc
        death_calc = 1 - healthy_calc - care_calc

        # make graph
        X = []
        for i in range(group_num):
            X.append(str(i*5) + '\u2013' + str(i*5+4))
        X = np.array(X)
        plt.figure(figsize=(6, 4.5))
        plt.plot(X, care_disease_end, color='#000000', marker='.', ls='', label='diseaese_measured')
        plt.plot(X, care_disease_calc, color='#000000', ls='-', label='disease_simulated')
        plt.xlabel('Age', fontsize=font_size)
        plt.ylabel('Prevalence of disease $x^{\\theta_j}_{i, \mathrm{C}}$  [\u2013]', fontsize=font_size)
        plt.xticks(np.arange(len(X)), rotation=90)
        plt.tick_params(labelsize=font_size)
        plt.title(f'{start_year}\u2013{end_year}', fontsize=font_size)
        plt.ylim([0, ymax_suggested])
        plt.savefig(os.path.join(output_dir, f'x_{disease_name[:2]}_calc_{end_year}_from_{start_year}.tif'), bbox_inches='tight', dpi=300)
        plt.show()

        # output fitting result to csv
        df_calc = df_end_disease.copy()
        df_calc.iloc[0:group_num, [2]] = alive_not_disease_calc
        df_calc.iloc[0:group_num, [3]] = care_disease_calc
        df_calc.iloc[0:group_num, [4]] = death_calc
        df_calc.drop(index=df_calc.index[group_num:], inplace=True)
        df_calc.to_csv(os.path.join(output_dir, f'X_{disease_name[:2]}_calc_{end_year}_from_{start_year}.csv'), index=False)
        
    
if __name__ == '__main__':
    file_path_disease_name = os.path.join(input_dir, '00_basic', 'disease_name.csv')
    df_disease_name = pd.read_csv(file_path_disease_name)
    disease_folder = df_disease_name.iloc[:, 0].tolist()
    disease_names = []
    # record_beta = []
    for disease in disease_folder:
        each_disease_calc(disease)
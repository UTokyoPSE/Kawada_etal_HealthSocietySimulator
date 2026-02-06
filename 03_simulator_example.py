"""
Source code attached for the manuscript:

"Development of a health care system simulator for Japan toward efficient allocation of medical resources" by
Yusuke Kawada, Yusuke Hayashi, and Hirokazu Sugiyama

This code corresponds to the simulator & Case studies shown in the manuscript.
Details are described in "Methods / Simulation module module", 
    "Results / Case Study 1: Increasing the healthy population".
    and "Results / Case Study 2: Decreased death".
The output figure corresponds to "Supplementary Information S1.

Created by Yusuke Kawada/Sugiyama-Badr Lab/The University of Tokyo
Last saved on Feb 05 2026
"""

import os
import numpy as np
from datetime import datetime
from simulator.a0_simulation_main import Context, InitialCondition, SimulationWhole
from simulator.b0_data_process_whole import data_process, data_process_for_case_study

iter_num = 20  # Number of simulations for each scenario
ctx = Context()

def simulation_with_beta_change(target_disease_indexs, output_dir, downratio=0.6, iter_num=20):
    for simu_type in range(1, 3):
        print(f"---β×{downratio}: Assumption {simu_type}---")
        # Set initial conditions
        initial_condition = InitialCondition(ctx, simu_type) 
        # Create class to manage entire simulation
        simulation_whole = SimulationWhole(ctx, initial_condition, output_dir)

        # Modify beta (keep negative values unchanged, multiply positive values by downratio)
        beta_base = simulation_whole.disease_classes[target_disease_indexs].array_of_beta
        beta_scenario = np.where(beta_base > 0, beta_base * downratio, beta_base)
        simulation_whole.disease_classes[target_disease_indexs].array_of_beta = beta_scenario
        
        for i_simu in range(iter_num):
            simulation_whole.new_simulation()
            simulation_whole.simulation_conduct()
            print(f'Simulation {i_simu+1} completed')

if __name__ == "__main__":
    now = datetime.now()
    now = now.strftime('%y%m%d%H%M')
    output_base_dir = os.path.join('output', 'For_simulator')
    output_raw_dir = os.path.join(output_base_dir, f'00_Raw_result_{now}')  # Create directory to save simulation results
    os.makedirs(output_raw_dir, exist_ok=True)

# simulation for BaseCase-------------------------------------------------------------------------------------------
    output_dir = os.path.join(output_raw_dir, '00_Basecase')
    os.makedirs(output_dir, exist_ok=True)  # Create output directory
    for simu_type in range(1, 3):
        print(f'---Basecase simulation: Assumption {simu_type}---')
        initial_condition = InitialCondition(ctx, simu_type) 
        simulation_whole = SimulationWhole(ctx, initial_condition, output_dir)

        for i_simu in range(iter_num):
            simulation_whole.new_simulation()
            simulation_whole.simulation_conduct()
            print(f'Simulation {i_simu+1} completed')

# simulation for Case Studies: Decrease beta for target disease------------------------------------------------------
    for target_disease_index in range(len(ctx.disease_list)):
        target_disease = ctx.disease_list[target_disease_index]
        print(f"------simulation for {target_disease}------")
        downratio = 0.6
        output_dir = os.path.join(output_raw_dir, f"{target_disease_index+1:02}_{target_disease}_beta_{downratio:.1f}")
        os.makedirs(output_dir, exist_ok=True)
        simulation_with_beta_change(target_disease_index, output_dir, downratio=downratio, iter_num=iter_num)

    data_process(output_base_dir, now, iter_num=iter_num)
    data_process_for_case_study(output_base_dir, now)
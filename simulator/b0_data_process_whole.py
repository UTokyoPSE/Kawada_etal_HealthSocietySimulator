from simulator import b4_data_extract as data_extract
from simulator import b5_compare_disease as compare_disease
import os
import re
import numpy as np
import pandas as pd

def aggregate_results(dir_name: str):
    '''
    Aggregate repeated simulation results by disease
    CSV files containing repeated simulation results for each health state and disease are saved in folders for each year
    The aggregated results are saved in the Summary folder
    '''
    folders = sorted([f for f in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, f))])
    folders = [folder for folder in folders if re.match(r'^\d+$', folder) is not None]
    # Remove empty folders
    for folder in reversed(folders):
        if len([f for f in os.listdir(os.path.join(dir_name, folder)) if f.endswith('.csv')]) == 0:
            folders.remove(folder)
            remove_dir = os.path.join(dir_name, folder)
            os.rmdir(remove_dir)

    file_names = [f for f in os.listdir(os.path.join(dir_name, folders[0])) if f.endswith('.csv')]
    for file_name in file_names:
        # Exclude files related to initial state, only files in '(year).csv' format remain
        if re.match(r'\d{4}.csv', file_name) is None:
            file_names.remove(file_name)

    for file_name in file_names:
        year = re.match(r'\d{4}', file_name).group()
        for simu_type in range(2):  # Aggregate two simulation results: Assumption 1and Assumption 2
            dfs = []
            mid_point = len(folders)//2
            for i_folder in range(simu_type * mid_point, (simu_type + 1) * mid_point):
                file_path = os.path.join(dir_name, folders[i_folder], file_name)
                df = pd.read_csv(file_path, index_col=0)
                index_name = df.index
                for i_row in range(len(df.index)):
                    try:
                        dfs[i_row].loc[len(dfs[i_row])] = df.iloc[i_row, :]   
                    except:
                        dfs.append(pd.DataFrame(columns=df.columns))
                        dfs[i_row].loc[len(dfs[i_row])] = df.iloc[i_row, :]

            # Save to CSV file    
            for i_row in range(len(dfs)):
                output_dir = os.path.join(dir_name, 'Aggregated', f'{year}_Assumption{simu_type+1}')
                os.makedirs(output_dir, exist_ok=True)
                if i_row - 3 <= 0:
                    index_num = f'00_{i_row}'
                else:
                    index_num = f'{i_row-3:02}'
                dfs[i_row].to_csv(os.path.join(output_dir, f'{index_num}_{index_name[i_row]}.csv'), index=False)

def summarize_results(raw_dir: str, aver_dir: str, iter_num = 20):
    '''
    Use data converted by csv_process.py
    Calculate mean and standard deviation for each initial value type, and plot on graphs
    Also compare with actual data in intervals where actual data is available
    '''
    dir_name = os.path.join(raw_dir, 'Aggregated')
    
    folders = sorted([f for f in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, f))])
    file_names = [f for f in os.listdir(os.path.join(dir_name, folders[0])) if f.endswith('.csv')]
    for i_year in range(len(folders)//2): #(1, len(folders)//2):
        # Load demographic data
        demography = pd.read_csv(os.path.join(dir_name, folders[2 * i_year], file_names[0]))
        demography = demography.iloc[0, :]

        # Calculate mean and standard deviation for each disease (Type 1)
        df_mean_type1 = pd.DataFrame(columns=demography.index)
        df_std_type1 = pd.DataFrame(columns=demography.index)
        for file_name in file_names:
            file_path = os.path.join(dir_name, folders[2 * i_year], file_name)
            df = pd.read_csv(file_path)
            mean = df.mean()
            std = df.std(ddof=1)  # Sample standard deviation
            ste = std / np.sqrt(iter_num)  # Standard error
            index_name = re.sub(r'.csv', '', file_name)
            df_mean_type1.loc[index_name] = mean
            df_std_type1.loc[index_name] = ste
        
        # Calculate mean and standard deviation for each disease (Type 2)    
        df_mean_type2 = pd.DataFrame(columns=demography.index)
        df_std_type2 = pd.DataFrame(columns=demography.index)
        for file_name in file_names:
            file_path = os.path.join(dir_name, folders[2 * i_year+1], file_name)
            df = pd.read_csv(file_path)
            mean = df.mean()
            std = df.std(ddof=1)
            ste = std / np.sqrt(iter_num)
            index_name = re.sub(r'.csv', '', file_name)
            df_mean_type2.loc[index_name] = mean
            df_std_type2.loc[index_name] = ste
        
        year = re.match(r'\d{4}', folders[2 * i_year]).group()
        
        # Save to CSV
        result_dir = os.path.join(aver_dir, year)
        os.makedirs(result_dir, exist_ok=True)
        df_mean_type1.to_csv(os.path.join(result_dir, 'Assumption1_mean.csv'))
        df_std_type1.to_csv(os.path.join(result_dir, 'Assumption1_ste.csv'))
        df_mean_type2.to_csv(os.path.join(result_dir, 'Assumption2_mean.csv'))
        df_std_type2.to_csv(os.path.join(result_dir, 'Assumption2_ste.csv'))

def comparison_with_basecase(ave_dir:str, diff_dir:str, base_dir:str): # シミュレーションの回数
    '''
    calculate the difference between the average results and the basecase results for each year and each assumption
    '''
    target_files = [['Assumption1_mean.csv', 'Assumption1_ste.csv'], ['Assumption2_mean.csv', 'Assumption2_ste.csv']]
    year_folders = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]

    for year_folder in year_folders:
        result_dir = os.path.join(diff_dir, year_folder)
        base_y_dir = os.path.join(base_dir, year_folder)
        ave_y_dir = os.path.join(ave_dir, year_folder)
        os.makedirs(result_dir, exist_ok=True)
        # Load data for each Assumption1 and Assumption2
        for target_file in target_files:
            # Load mean and standard deviation data
            df_mean = pd.read_csv(os.path.join(ave_y_dir, target_file[0]), index_col=0)
            df_std = pd.read_csv(os.path.join(ave_y_dir, target_file[1]), index_col=0)
            df_mean_base = pd.read_csv(os.path.join(base_y_dir, target_file[0]), index_col=0)
            df_std_base = pd.read_csv(os.path.join(base_y_dir, target_file[1]), index_col=0)
            
            demographic_row = df_mean.index.tolist()[0]
            other_rows = df_mean.index.tolist()[1:]

            df_demographic = df_mean.loc[demographic_row]
            df_mean = df_mean.loc[other_rows]
            df_std = df_std.loc[other_rows]
            df_mean_base = df_mean_base.loc[other_rows]
            df_std_base = df_std_base.loc[other_rows]

            # Calculate the difference
            df_mean_diff = df_mean - df_mean_base
            df_std_diff = np.sqrt((df_std**2 + df_std_base**2))
            # Save to CSV
            df_demographic.to_csv(os.path.join(result_dir, 'demographic.csv'), index=True)
            df_mean_diff.to_csv(os.path.join(result_dir, target_file[0]), index=True)
            df_std_diff.to_csv(os.path.join(result_dir, target_file[1]), index=True)

def convert_fraction_into_number(diff_dir:str, diff_num_dir:str): # シミュレーションの回数
    '''
    Records how the overall health status and the number of patients for each disease change relative to the baseline for each year, disease type, beta, and mu.
    By recording patient numbers instead of ratios, we can understand the actual increase or decrease in patient numbers.
    In ver2, also records the sum of x_h and x_c -> can be made compatible with regression equations created previously.
    '''
    target_files = [['Assumption1_mean.csv', 'Assumption1_ste.csv'], ['Assumption2_mean.csv', 'Assumption2_ste.csv']]
    year_folders = [name for name in os.listdir(diff_dir) if os.path.isdir(os.path.join(diff_dir, name))]
    # Get data for each year
    for year_folder in year_folders:
        result_dir = os.path.join(diff_num_dir, year_folder)
        diff_y_dir = os.path.join(diff_dir, year_folder)
        demographic_this_year = pd.read_csv(os.path.join(diff_y_dir, 'demographic.csv'), index_col=0)
        os.makedirs(result_dir, exist_ok=True)
        # Get data for each Assumption1 and Assumption2
        for target_file in target_files:
            # Load mean and standard deviation data
            df_mean = pd.read_csv(os.path.join(diff_y_dir, target_file[0]), index_col=0)
            df_std = pd.read_csv(os.path.join(diff_y_dir, target_file[1]), index_col=0)
            df_pop = df_mean.copy()
            for i_row in range(len(df_mean.index)):
                df_pop.iloc[i_row, :] = demographic_this_year.values.flatten()
            
            # Convert to patient numbers
            df_patient_num = df_mean * df_pop
            df_patient_num_std = df_std * df_pop

            # Add sum column
            df_patient_num['SUM'] = df_patient_num.sum(axis=1)
            df_patient_num_std['SUM'] = np.sqrt((df_patient_num_std ** 2).sum(axis=1))

            df_patient_num.to_csv(os.path.join(result_dir, target_file[0]), index=True)
            df_patient_num_std.to_csv(os.path.join(result_dir, target_file[1]), index=True)

def aggregate_difference_number(diff_num_base_dir:str, diff_agg_dir:str):
    '''
    Aggregate the data of increases and decreases in the number of people in health, care, and death status for each scenario into a single file
    '''
    target_files = [['Assumption1_mean.csv', 'Assumption1_ste.csv'], ['Assumption2_mean.csv', 'Assumption2_ste.csv']]
    disease_dirs = [name for name in os.listdir(diff_num_base_dir) if os.path.isdir(os.path.join(diff_num_base_dir, name))]
    in_dir = os.path.join(diff_num_base_dir, disease_dirs[0])
    year_folders = [name for name in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, name))]
    
    for year_folder in year_folders:
        result_dir = os.path.join(diff_agg_dir, year_folder)
        os.makedirs(result_dir, exist_ok=True)
        for target_file in target_files:
            indicator_name = ['N_health', 'N_care', 'N_death']
            for i_indicator in range(len(indicator_name)):
                df_indicator_mean = pd.DataFrame()
                df_indicator_std = pd.DataFrame()
                for disease_dir in disease_dirs:
                    diff_num_dir = os.path.join(diff_num_base_dir, disease_dir, year_folder)
                    df_disease_mean = pd.read_csv(os.path.join(diff_num_dir, target_file[0]), index_col=0)
                    df_disease_std = pd.read_csv(os.path.join(diff_num_dir, target_file[1]), index_col=0)
                    df_disease_mean = df_disease_mean.iloc[i_indicator, :]
                    df_disease_std = df_disease_std.iloc[i_indicator, :]
                    
                    df_indicator_mean[disease_dir] = df_disease_mean
                    df_indicator_std[disease_dir] = df_disease_std
                
                # Save to CSV
                df_indicator_mean = df_indicator_mean.T
                df_indicator_std = df_indicator_std.T
                file_name = f'{str(i_indicator).zfill(2)}_{indicator_name[i_indicator]}_{target_file[0]}'
                df_indicator_mean.to_csv(os.path.join(result_dir, file_name), index=True)
                file_name = f'{str(i_indicator).zfill(2)}_{indicator_name[i_indicator]}_{target_file[1]}'
                df_indicator_std.to_csv(os.path.join(result_dir, file_name), index=True)

def output_for_case_study(diff_agg_dir:str, case_dir:str):
    os.makedirs(case_dir, exist_ok=True)
    year_folders = [f for f in os.listdir(diff_agg_dir) if os.path.isdir(os.path.join(diff_agg_dir, f))]
    df_cost = pd.read_csv(os.path.join('input', 'For_Simulator', 'Cost_proxy.csv'), index_col=0)
    target_files = ['Assumption1_mean.csv', 'Assumption1_ste.csv', 'Assumption2_mean.csv', 'Assumption2_ste.csv']
    file_names = [f for f in os.listdir(os.path.join(diff_agg_dir, year_folders[0])) if f.endswith('.csv')]
    for i_name in range(len(file_names)):
        for target_file in target_files:
            file_names[i_name] = re.sub(rf'_{target_file}', '', file_names[i_name])

    file_names = sorted(list(set(file_names)))
    for year_folder in reversed(year_folders):
        diff_agg_y_dir = os.path.join(diff_agg_dir, year_folder)
        df_mean = pd.DataFrame()
        df_std = pd.DataFrame()
        df_mean['cost_mean'] = df_cost['beta_mean']
        df_std['cost_std'] = df_cost['beta_std']
        for file_name in file_names:
            df_av_1 = pd.read_csv(os.path.join(diff_agg_y_dir, f'{file_name}_Assumption1_mean.csv'), index_col=0)
            df_av_2 = pd.read_csv(os.path.join(diff_agg_y_dir, f'{file_name}_Assumption2_mean.csv'), index_col=0)
            df_ste_1 = pd.read_csv(os.path.join(diff_agg_y_dir, f'{file_name}_Assumption1_ste.csv'), index_col=0)
            df_ste_2 = pd.read_csv(os.path.join(diff_agg_y_dir, f'{file_name}_Assumption2_ste.csv'), index_col=0)
            df_mean[file_name] = (df_av_1['SUM'].values.flatten() + df_av_2['SUM'].values.flatten())/2
            df_std[file_name] = np.sqrt((df_ste_1['SUM'].values.flatten()**2 + df_ste_2['SUM'].values.flatten()**2)/2)
        df_result = pd.concat([df_mean, df_std], axis=1)
        df_result.to_csv(os.path.join(case_dir, f'{year_folder}.csv'), index=True)
        
def data_process(output_base_dir, simu_time, iter_num=20):
    output_raw_dir = os.path.join(output_base_dir, f'00_Raw_result_{simu_time}')
    output_aver_dir = os.path.join(output_base_dir, f'01_Average_result_{simu_time}')
    output_diff_dir = os.path.join(output_base_dir, f'02_Difference_from_base_result_{simu_time}')
    folders_in_mother = sorted([f for f in os.listdir(output_raw_dir) if (os.path.isdir(os.path.join(output_raw_dir, f))) and (re.match(r'^\d{2}', f) is not None)])
    total_folders = len(folders_in_mother)

    # ---------------------------------------------------------------------------
    print('aggregating the results...')
    for i, target_folder in enumerate(folders_in_mother):
        if i % (total_folders // 20) == 0: 
            print(f'Progress: {i / total_folders * 100:.1f}%')
        raw_dir = os.path.join(output_raw_dir, target_folder)
        ave_dir = os.path.join(output_aver_dir, target_folder)
        aggregate_results(raw_dir)
        summarize_results(raw_dir=raw_dir, aver_dir=ave_dir, iter_num=iter_num)
    print('Progress: 100.0%')
    # ---------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    print('calculating differences with base case...')
    base_dir = os.path.join(output_aver_dir, '00_Basecase')
    for i, target_folder in enumerate(folders_in_mother[1:]):
        if i % (total_folders // 20) == 0: 
            print(f'Progress: {i / total_folders * 100:.1f}%')
        ave_dir = os.path.join(output_aver_dir, target_folder)
        diff_dir = os.path.join(output_diff_dir, target_folder)
        comparison_with_basecase(ave_dir=ave_dir, diff_dir=diff_dir, base_dir=base_dir)
    print('Progress: 100.0%')
    # ----------------------------------------------------------------------------

def data_process_for_case_study(output_base_dir, simu_time):
    output_diff_dir = os.path.join(output_base_dir, f'02_Difference_from_base_result_{simu_time}')
    output_diff_num_dir = os.path.join(output_base_dir, f'03_Difference_number_from_base_result_{simu_time}')
    output_diff_agg_dir = os.path.join(output_base_dir, f'04_Difference_aggregated_{simu_time}')
    output_case_dir = os.path.join(output_base_dir, f'05_Case_study_result_{simu_time}')
    folders_in_mother = sorted([f for f in os.listdir(output_diff_dir) if (os.path.isdir(os.path.join(output_diff_dir, f))) and (re.match(r'^\d{2}', f) is not None)])
    total_folders = len(folders_in_mother)
    for i, target_folder in enumerate(folders_in_mother):
        if i % (total_folders // 20) == 0: 
            print(f'Progress: {i / total_folders * 100:.1f}%')
        diff_dir = os.path.join(output_diff_dir, target_folder)
        diff_num_dir = os.path.join(output_diff_num_dir, target_folder)
        convert_fraction_into_number(diff_dir=diff_dir, diff_num_dir=diff_num_dir)
    
    print('Progress: 100.0%')
    print('aggregating difference numbers...')
    aggregate_difference_number(diff_num_base_dir=output_diff_num_dir, diff_agg_dir=output_diff_agg_dir)
    output_for_case_study(diff_agg_dir=output_diff_agg_dir, case_dir=output_case_dir)

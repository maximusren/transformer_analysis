from utils import *
import streamlit as st
import base64
import io
from matplotlib.backends.backend_pdf import PdfPages
#%% Faa calculation
def calculate_Faa(TH):
    Faa = pow(np.e, 15000/383 - 15000/(273+TH))
    return Faa

#%% Thermal simulation 
def TRF_Thermal_Sim(TA0, P, deltaTTO_R, deltaTH_R, R, Tc, Tcw, n, m, kVA, ambient_temperature, df_transformer, pf, dt, Q, V):
    
    # Interpolation
    interp_indices = np.arange(0, len(ambient_temperature), dt)
    ambtemp = np.interp(interp_indices, np.arange(len(ambient_temperature)), ambient_temperature['Tamb (C)'].values)
    power = np.interp(interp_indices, np.arange(len(df_transformer)), df_transformer['kwh_per_interval'].values)
    voltage = np.interp(interp_indices, np.arange(len(df_transformer)), df_transformer['voltage'].values)

    # Pre-allocate arrays
    n_steps = len(interp_indices)
    TO_temp = np.zeros(n_steps)
    TH_temp = np.zeros(n_steps)
    Faa_list = np.zeros(n_steps)
    
    # Initialize
    TTO = TA0
    TH = TA0

    # Calculations
    S = np.sqrt(power**2 + Q**2) 
    I = S / (voltage * pf)        
    Vpu = voltage / V
    Ipu = I * V / kVA

    for i in range(n_steps):
        deltaTTO = deltaTTO_R * ((Ipu[i]**2 * R + Vpu[i]**2) / (R + 1))**n
        TTO_U = ambtemp[i] + deltaTTO
        TTO = (TTO_U - TTO) * (1 - np.exp(-dt/Tc)) + TTO
        
        deltaTH = deltaTH_R * (Ipu[i]**2)**m
        TH_U = TTO + deltaTH
        TH = (TH_U - TH) * (1 - np.exp(-dt/Tcw)) + TH
        
        TO_temp[i] = TTO
        TH_temp[i] = TH
        Faa_list[i] = np.exp(15000/383 - 15000/(273 + TH))

    return TO_temp, TH_temp, interp_indices, Faa_list

#%% Plot the results of the thermal analysis in Python

def plot_results2(transformer, top_oil_results, hot_spot_results, df_transformer, time, ambtemp, power, dt, meter_consumer_transformer): #transformer argument only provides the name
    fig = plt.figure()
    fig = plt.figure(figsize=(10, 12))
    
    # Make it 4 by 1
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)

    # Debug information
    # st.write(f"Debug plotting for transformer {transformer}:")
    # st.write(f"Length of df_transformer.index: {len(df_transformer.index)}")
    # st.write(f"dt value: {dt}")
    # st.write(f"Length of ambtemp: {len(ambtemp)}")
    # st.write(f"Length of power: {len(power)}")
    # st.write(f"Length of top_oil_results: {len(top_oil_results)}")
    # st.write(f"Length of hot_spot_results: {len(hot_spot_results)}")
    
    # Use the original dates from df_transformer
    dates = df_transformer.index
    
    if dt < 1:
        # Downsample the results to match the original time scale
        step = int(1/dt) 
        top_oil_results = top_oil_results.iloc[::step].reset_index(drop=True)
        hot_spot_results = hot_spot_results.iloc[::step].reset_index(drop=True)
        
        # Make sure doesn't exceed the original length
        min_length = min(len(dates), len(top_oil_results), len(hot_spot_results))
        dates = dates[:min_length]
        top_oil_results = top_oil_results[:min_length]
        hot_spot_results = hot_spot_results[:min_length]
    
    ax1.step(dates, ambtemp[:len(dates)], where='post')
    ax2.step(dates, power[:len(dates)], where='post')
    
    # Set limits
    ax1.set_xlim(dates[0], dates[-1])
    ax2.set_xlim(dates[0], dates[-1])
    ax3.set_xlim(dates[0], dates[-1])
    ax4.set_xlim(dates[0], dates[-1])
    
    ax2.set_ylim(0, 70)
    ax3.set_ylim(0, 225)
    ax4.set_ylim(0, 225)
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax1.set_title('Ambient Temperature', fontsize=20)
    ax1.set_ylabel("C", fontsize=16)
    ax2.set_title('Power', fontsize=20)
    ax2.set_ylabel("kW", fontsize=16)
    
    # Plot top oil results
    for col in top_oil_results.columns: 
        ax3.plot(dates, top_oil_results[col][:len(dates)], alpha=0.3, linewidth=0.5, color='gray')
    
    max_envelope = top_oil_results.max(axis=1)
    min_envelope = top_oil_results.min(axis=1)
    ax3.plot(dates, max_envelope[:len(dates)], alpha=1, linewidth=0.7, color='#962d3e')
    ax3.plot(dates, min_envelope[:len(dates)], alpha=1, linewidth=0.7, color='#962d3e')
    
    # Plot hot spot results
    for col in hot_spot_results.columns: 
        ax4.plot(dates, hot_spot_results[col][:len(dates)], alpha=0.3, linewidth=0.5, color='gray')
    
    max_envelope = hot_spot_results.max(axis=1)
    min_envelope = hot_spot_results.min(axis=1)
    ax4.plot(dates, max_envelope[:len(dates)], alpha=1, linewidth=0.7, color='#962d3e')
    ax4.plot(dates, min_envelope[:len(dates)], alpha=1, linewidth=0.7, color='#962d3e')
    
    ax3.set_title('Top Oil Temperature', fontsize=20)
    ax3.set_ylabel("C", fontsize=16)
    ax4.set_title('Hot Spot Temperature', fontsize=20)
    ax4.set_ylabel("C", fontsize=16)
    
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    fig.suptitle(f"Transformer: {transformer}, EVs: {get_transformer_ev_count(transformer, meter_consumer_transformer)}", fontsize=24)
    plt.tight_layout()
    
    return fig

#%% Monte-Carlo simulation

def doThermalAnalysis(transformer, ambient_temperature, meter_consumer_transformer, meter_reads, 
                     N_Runs, pf, dt, R, Tc, TA0, P, Q, V, S, I, deltaTTO_R, deltaTH_R, Tcw, kVA,
                     Vpu, Ipu, n, m,
                     dT_OR_min, dT_OR_max, dT_HR_min, dT_HR_max, 
                     R_C2I_min, R_C2I_max, tau_R_min, tau_R_max, 
                     tau_W_min, tau_W_max,
                     simulation_progress_bar=None,  
                     status_text=None,             
                     transformer_idx=1,          
                     total_transformers=1):    
    
    # Initial top oil and hot spot temperatures
    TTO = TA0
    TH = TA0

    # Clean ambient temperature data
    ambient_temperature.index = pd.to_datetime(ambient_temperature['datetime'])
    ambient_temperature = ambient_temperature.sort_index()
    ambient_temperature = ambient_temperature[~ambient_temperature.index.duplicated(keep='first')]
    
    df_transformer = pd.DataFrame(index=ambient_temperature.index)
    meters_under_transformer = get_meters(transformer, meter_consumer_transformer)
    
    df_transformer['voltage'] = 0
    df_transformer['kwh_per_interval'] = 0
    
    # Get kwh per interval for all the meters in a transformer & get average voltage
    voltage_dfs = []
    for meter_id in meters_under_transformer:
        df_meter = meter_reads[meter_reads['meter_id']==meter_id]
        
        # Find overlapping date range
        start_date = max(ambient_temperature.index.min(), df_meter.index.min())
        end_date = min(ambient_temperature.index.max(), df_meter.index.max())
        
        # Check if there are overlapping dates
        if start_date <= end_date:
            df_transformer[meter_id] = df_meter['kwh_per_interval'][start_date:end_date].resample('1h').sum()
            voltage_dfs.append(df_meter['voltage'][start_date:end_date].resample('1h').first())
            df_transformer['kwh_per_interval'] = df_transformer['kwh_per_interval'] + df_meter['kwh_per_interval'][start_date:end_date].resample('1h').sum()
        else:
            print(f"No overlapping dates found for meter {meter_id}")
            continue
        
    # Get average voltage
    combined_voltage_dfs = pd.concat(voltage_dfs, axis=1)   
    avg_voltages = combined_voltage_dfs.mean(axis=1, skipna=True)
        
    df_transformer['voltage'] = avg_voltages
    
    df_transformer['voltage'] = df_transformer['voltage'].replace(0, np.nan)
    df_transformer['voltage'] = df_transformer['voltage'].ffill()
    
    ambient_temperature = ambient_temperature.reindex(df_transformer.index)
    ambient_temperature = ambient_temperature.interpolate(method='linear')

    sim_result = []
    top_oil_results = {}
    hot_spot_results = {}
    Faa_results = {}
    yearly_loss = {}
    
    for irun in range(N_Runs):
        if simulation_progress_bar and status_text:
            progress = int(((irun + 1) / N_Runs) * 100)
            simulation_progress_bar.progress(progress)
            status_text.text(
                f"Transformer {transformer_idx}/{total_transformers} - {transformer}\n"
                f"Running simulation {irun+1}/{N_Runs} ({progress}%)"
            )
            
        kVA = get_kva(transformer, meter_consumer_transformer)
        
        if irun==0: # Worse case scenario
            R = R_C2I_max
            Tc = tau_R_min
            Tcw = tau_W_min
            deltaTTO_R = dT_OR_max
            deltaTH_R = dT_HR_min
        else:
            R = random.uniform(R_C2I_min, R_C2I_max)
            Tc = random.uniform(tau_R_min, tau_R_max)
            Tcw = random.uniform(tau_W_min, tau_W_max)   
            deltaTTO_R = random.uniform(dT_OR_min, dT_OR_max)
            deltaTH_R = random.uniform(dT_HR_min, dT_HR_max)
        
        # Do simulation and find out time series temperature (top oil temperature, and hot spot temperature)
        TO_temp, TH_temp, time, Faa_list = TRF_Thermal_Sim(
            # The following are input variables
            TA0,
            P,
            # The following are parameters
            deltaTTO_R,
            deltaTH_R,
            R,
            Tc,
            Tcw,
            n,
            m,
            kVA,
            ambient_temperature,
            df_transformer,
            pf,
            dt,
            Q,
            V,
            ) 
        
        top_oil_results[f'test_{len(sim_result)}'] = TO_temp
        hot_spot_results[f'test_{len(sim_result)}'] = TH_temp
        Faa_results[f'test_{len(sim_result)}'] = Faa_list
        yearly_loss[f'test_{len(sim_result)}'] = np.nansum(Faa_list) * dt
        
        # save result
        this_run = {
            'irun': irun,
            'dT_OR': deltaTTO_R,
            'dT_HR': deltaTH_R,
            'R_C2I': R,
            'tau_R': Tc,
            'tau_W': Tcw,
            'n': n,
            'm': m,
            'T_TO': TTO,
            'T_HS': TH,
            }
        sim_result.append(this_run)
    
    top_oil_results = pd.DataFrame(top_oil_results)
    hot_spot_results = pd.DataFrame(hot_spot_results)
    Faa_results = pd.DataFrame(Faa_results)
    
    return top_oil_results, hot_spot_results, ambient_temperature, df_transformer, time, Faa_results, yearly_loss, sim_result

#%%

def display_thermal_results(dt, meter_info_df, current_transformer_results, transformer_id):
    st.markdown(f"---") # Separator
    st.markdown(f"### {transformer_id} Thermal Analysis Results")
    with st.expander(f"View Details for {transformer_id}", expanded=True): # Expander is open by default
        # Generate and display plot as PDF
        fig = plot_results2(
            transformer_id,
            current_transformer_results['top_oil'],
            current_transformer_results['hot_spot'],
            current_transformer_results['transformer_data'],
            current_transformer_results['time'],
            current_transformer_results['ambient_temperature']['Tamb (C)'],
            current_transformer_results['transformer_data']['kwh_per_interval'],
            dt,
            meter_info_df # Pass meter_info_df for EV count
        )
        pdf_buffer = generate_pdf_report(fig, transformer_id) # Utility function
        base64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="700" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

        # Display summary metrics for the transformer
        st.subheader("Summary Metrics")
        col1_disp, col2_disp = st.columns(2)
        with col1_disp:
            st.metric("Worst Hourly Aging", f"{current_transformer_results['metrics']['max_hourly_aging']:.2f} hours", help="Maximum loss of life in any single hour (per unit)")
            st.metric("Yearly Aging Impact", f"{current_transformer_results['metrics']['yearly_aging_hours']:.1f} hours", help="Equivalent hours of aging per year")
        with col2_disp:
            st.metric("Peak Hot Spot", f"{current_transformer_results['metrics']['max_hot_spot']:.1f}°C", help="Maximum winding hot spot temperature")
            st.metric("Peak Top Oil", f"{current_transformer_results['metrics']['max_top_oil']:.1f}°C", help="Maximum top oil temperature")

        # Display simulation parameters used
        st.subheader("Simulation Parameters (Monte Carlo Runs)")
        sim_df = pd.DataFrame(current_transformer_results['params'])
        st.dataframe(sim_df.style.format(precision=2))

        # Download button for detailed results of this transformer
        combined_data = create_download_data(transformer_id, current_transformer_results) # Utility function
        csv_combined = combined_data.to_csv(index=True).encode('utf-8')
        st.download_button(
            label=f"Download All Results for {transformer_id} (CSV)",
            data=csv_combined,
            file_name=f"{transformer_id}_thermal_analysis_results.csv",
            mime='text/csv',
            key=f"download_csv_{transformer_id}"
        )
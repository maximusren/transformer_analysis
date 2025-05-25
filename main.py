import streamlit as st
from utils import * # Utility functions
from algorithm import * # Core thermal analysis algorithm
#%%
# Initialize session state variables to store data and application state
if 'meter_reads_df' not in st.session_state:
    st.session_state.meter_reads_df = None # Stores processed meter reading data
if 'meter_info_df' not in st.session_state:
    st.session_state.meter_info_df = None # Stores processed meter information data
if 'ambient_temperature_df' not in st.session_state:
    st.session_state.ambient_temperature_df = None # Stores processed ambient temperature data

if 'files_uploaded' not in st.session_state:
    # Tracks the upload status of required files
    st.session_state.files_uploaded = {
        'meter_reads': False,
        'meter_information': False,
        'ambient_temperature_data': False
    }

if 'all_results_cache' not in st.session_state:
    st.session_state.all_results_cache = {} # Caches results of the thermal analysis
if 'current_transformer' not in st.session_state:
    st.session_state.current_transformer = None # Stores the ID of the transformer currently being processed

if 'action' not in st.session_state:
    st.session_state.action = None # Stores the action that the user has chosen
if 'thermal_analysis_completed' not in st.session_state:
    st.session_state.thermal_analysis_completed = False # Stores whether the thermal analysis has been completed

if 'dt' not in st.session_state:
    st.session_state.dt = None
#%% 
    
# Configure Streamlit page settings
st.set_page_config(layout="wide")

st.title('Transformer Thermal Analysis Tool')
st.write('Upload your data and run thermal analysis on transformers.')

# --- File Upload Section ---
st.header("Upload Data Files")

# Upload meter reads data
uploaded_meter_reads_file = st.file_uploader(
    "Upload Meter Data File (.csv)",
    type=['csv'],
    key="meter_reads_uploader",
    help="CSV file containing Meter ID, Timestamp, kWh per Interval, Voltage"
)

if uploaded_meter_reads_file is not None:
    try:
        meter_reads_raw = pd.read_csv(uploaded_meter_reads_file)
        st.success("Meter Reads file uploaded successfully!")
        # Layout for mapping columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Raw Meter Reads (Preview)")
            st.dataframe(meter_reads_raw.head())
        with col2:
            st.subheader("Map Your Columns")
            available_cols = meter_reads_raw.columns.tolist()
            # User selects columns corresponding to required data fields
            meter_id_col = st.selectbox("Meter ID", available_cols, index=available_cols.index('meter_id') if 'meter_id' in available_cols else 0, key="map_meter_id")
            timestamp_col = st.selectbox("Timestamp", available_cols, index=available_cols.index('DATATIMESTAMP') if 'DATATIMESTAMP' in available_cols else 0, key="map_timestamp")
            kwh_col = st.selectbox("kWh per Interval", available_cols, index=available_cols.index('kwh_per_interval') if 'kwh_per_interval' in available_cols else 0, key="map_kwh")
            voltage_col = st.selectbox("Voltage", available_cols, index=available_cols.index('voltage') if 'voltage' in available_cols else 0, key="map_voltage")
        with col3:
            st.subheader("Mapped Meter Reads (Preview)")
            # Create a standardized DataFrame
            meter_reads = pd.DataFrame({
                  'meter_id': meter_reads_raw[meter_id_col],
                  'DATATIMESTAMP': meter_reads_raw[timestamp_col],
                  'kwh_per_interval': meter_reads_raw[kwh_col],
                  'voltage': meter_reads_raw[voltage_col],
              })
            st.dataframe(meter_reads.head())
        # Process and store in session state
        meter_reads['time'] = pd.to_datetime(meter_reads['DATATIMESTAMP']) # Ensure correct format, adjust if needed
        meter_reads.set_index('time', inplace=True)
        st.session_state.meter_reads_df = meter_reads
        st.session_state.files_uploaded['meter_reads'] = True
    except Exception as e:
        st.error(f"Error processing Meter Reads file: {e}")
        st.session_state.meter_reads_df = None
        st.session_state.files_uploaded['meter_reads'] = False
else:
    # Reset upload status if file is removed or not uploaded
    st.session_state.files_uploaded['meter_reads'] = False if st.session_state.meter_reads_df is None else True

# Upload meter information data
uploaded_meter_info_file = st.file_uploader(
    "Upload Meter Information File (.csv)",
    type=['csv'],
    key="meter_info_uploader",
    help="CSV with Meter ID, Consumer ID, Transformer ID, Type, Multiplier, Known EV"
)
if uploaded_meter_info_file is not None:
    try:
        meter_info_raw = pd.read_csv(uploaded_meter_info_file)
        st.success("Meter Information file uploaded successfully!")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Raw Meter Information (Preview)")
            st.dataframe(meter_info_raw.head())
        with col2:
            st.subheader("Map Your Columns")
            available_cols = meter_info_raw.columns.tolist()
            meter_id_col = st.selectbox("Meter ID", available_cols, index=available_cols.index('meter_id') if 'meter_id' in available_cols else 0, key="map_info_meter_id")
            consumer_id_col = st.selectbox("Consumer ID", available_cols, index=available_cols.index('consumer_id') if 'consumer_id' in available_cols else 0, key="map_info_consumer_id")
            transformer_id_col = st.selectbox("Transformer ID", available_cols, index=available_cols.index('transformer_id') if 'transformer_id' in available_cols else 0, key="map_info_transformer_id")
            transformer_type_col = st.selectbox("Transformer Type", available_cols, index=available_cols.index('transformer_type') if 'transformer_type' in available_cols else 0, key="map_info_transformer_type")
            multiplier_col = st.selectbox("Meter Read Multiplier", available_cols, index=available_cols.index('meter_read_multiplier') if 'meter_read_multiplier' in available_cols else 0, key="map_info_multiplier")
            known_ev_col = st.selectbox("Known EV", available_cols, index=available_cols.index('known_ev') if 'known_ev' in available_cols else 0, key="map_info_known_ev")
        with col3:
            st.subheader("Mapped Meter Information (Preview)")
            meter_consumer_transformer = pd.DataFrame({
                'meter_id': meter_info_raw[meter_id_col],
                'consumer_id': meter_info_raw[consumer_id_col],
                'transformer_id': meter_info_raw[transformer_id_col],
                'transformer_type': meter_info_raw[transformer_type_col],
                'meter_read_multiplier': meter_info_raw[multiplier_col],
                'known_ev': meter_info_raw[known_ev_col],
            })
            st.dataframe(meter_consumer_transformer.head())
        # Extract kVA rating and store in session state
        meter_consumer_transformer['kva_rating'] = meter_consumer_transformer['transformer_type'].apply(extract_kva)
        st.session_state.meter_info_df = meter_consumer_transformer
        st.session_state.files_uploaded['meter_information'] = True
    except Exception as e:
        st.error(f"Error processing Meter Information file: {e}")
        st.session_state.meter_info_df = None
        st.session_state.files_uploaded['meter_information'] = False
else:
    st.session_state.files_uploaded['meter_information'] = False if st.session_state.meter_info_df is None else True

# Upload ambient temperature data
uploaded_ambient_temperature_file = st.file_uploader(
    "Upload Ambient Temperature Data (.csv or .pkl)",
    type=['csv', 'pkl'],
    key="ambient_temperature_uploader",
    help="CSV or Pickle file with Timestamp and Ambient Temperature (C)"
)
if uploaded_ambient_temperature_file is not None:
    try:
        ambient_temperature_raw = None
        file_name = uploaded_ambient_temperature_file.name
        # Read based on file extension
        if file_name.lower().endswith('.csv'):
            ambient_temperature_raw = pd.read_csv(uploaded_ambient_temperature_file)
        elif file_name.lower().endswith('.pkl'):
            ambient_temperature_raw = pd.read_pickle(uploaded_ambient_temperature_file)
            if isinstance(ambient_temperature_raw, pd.Series): # Convert Series to DataFrame if pkl is a Series
                ambient_temperature_raw = ambient_temperature_raw.to_frame().reset_index()
        else:
            st.error("Unsupported file type for Ambient Temperature. Please use .csv or .pkl")
            raise ValueError("Unsupported file type")

        st.success("Ambient Temperature file uploaded successfully!")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Raw Ambient Temperature (Preview)")
            st.dataframe(ambient_temperature_raw.head())
        with col2:
            st.subheader("Map Your Columns")
            available_cols = ambient_temperature_raw.columns.tolist()
            datetime_col = st.selectbox("Timestamp", available_cols, index=available_cols.index('datetime') if 'datetime' in available_cols else 0, key="map_amb_datetime")
            tamb_col = st.selectbox("Ambient Temperature (C)", available_cols, index=available_cols.index('Tamb (C)') if 'Tamb (C)' in available_cols else 0, key="map_amb_temp")
        with col3:
            st.subheader("Mapped Ambient Temperature (Preview)")
            ambient_temperature = pd.DataFrame({
                'datetime': ambient_temperature_raw[datetime_col],
                'Tamb (C)': ambient_temperature_raw[tamb_col],
            })
            st.dataframe(ambient_temperature.head())
        # Store in session state
        st.session_state.ambient_temperature_df = ambient_temperature
        st.session_state.files_uploaded['ambient_temperature_data'] = True
    except Exception as e:
        st.error(f"Error processing Ambient Temperature file: {e}")
        st.session_state.ambient_temperature_df = None
        st.session_state.files_uploaded['ambient_temperature_data'] = False
else:
    st.session_state.files_uploaded['ambient_temperature_data'] = False if st.session_state.ambient_temperature_df is None else True
#%%

# --- Thermal Analysis Section ---
st.markdown("---")
st.header("Transformer Analysis")

# Check if all required files are uploaded before proceeding
required_files = ['meter_reads', 'meter_information', 'ambient_temperature_data']
if not all(st.session_state.files_uploaded[file] for file in required_files):
    st.warning("Please upload all required data files to proceed with the analysis.")
    st.stop() # Stop execution if files are missing

thermal_analysis_completed = st.session_state.thermal_analysis_completed

# User action selection
action = st.selectbox("Choose an action", ["Select Action", "Thermal Analysis", "Raw Data Analysis"])
if action == "Thermal Analysis":
    if thermal_analysis_completed == True:
        display_results_or_rerun = st.selectbox("Would you like to display results or rerun thermal analysis?", ["Select Action", "Display Results", "Rerun Thermal Analysis"])
        if display_results_or_rerun == "Display Results":
            
            # Later: make it so display results displays the parameters as well (min and max values)
            
            
            results_display_area = st.container() # Container to display results incrementally
            dt = st.session_state.dt
            meter_info_df = st.session_state.meter_info_df
            
            for transformer_id in st.session_state.all_results_cache.keys():
                current_transformer_results = st.session_state.all_results_cache[transformer_id]
                
                
                with results_display_area: # Use the pre-defined container
                    display_thermal_results(dt, meter_info_df, current_transformer_results, transformer_id)
                        
            # Summary of results 
            st.markdown("---")
            st.header("Overall Analysis Summary")
            
            all_results = st.session_state.all_results_cache
            meter_info_df_summary = st.session_state.meter_info_df # Use for EV count
    
            if all_results: # Check if there are any results to summarize
                processed_transformer_ids = list(all_results.keys())
                total_evs = sum(get_transformer_ev_count(t, meter_info_df_summary) for t in processed_transformer_ids) # Utility function
    
                # Calculate overall peak hot spot temperature
                all_hot_spot_maxes = [
                    r['metrics']['max_hot_spot'] for r in all_results.values()
                    if r['metrics']['max_hot_spot'] is not None and pd.notna(r['metrics']['max_hot_spot'])
                ]
                max_temp_overall = max(all_hot_spot_maxes) if all_hot_spot_maxes else 0.0
    
                # Display overall metrics
                col1_sum, col2_sum, col3_sum = st.columns(3)
                with col1_sum:
                    st.metric("Total Transformers Processed", len(all_results))
                with col2_sum:
                    st.metric("Total EVs in System (for processed transformers)", total_evs)
                with col3_sum:
                    st.metric("Peak Hot Spot Temp (Overall)", f"{max_temp_overall:.1f}°C")
    
                # Download button for summary metrics of all processed transformers
                st.download_button(
                    label="Download Overall Summary Metrics (CSV)",
                    data=convert_results_to_csv(all_results).encode('utf-8'), # Utility function
                    file_name="all_transformers_metrics_summary.csv",
                    mime="text/csv",
                    key="download_summary_csv_overall"
                ) 

            
            pass
        if display_results_or_rerun == "Rerun Thermal Analysis":
            thermal_analysis_completed = False
            pass
            
    if thermal_analysis_completed == False:
        st.subheader("Thermal Analysis Settings")
    
        # Define default and user-configurable parameters for the thermal model
        pf = st.slider("Power factor (default: 1.0)", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
        dt = 0.1 # Time step for simulation (fixed)
        # Default parameters (can be made configurable if needed)
        R_param = 5.5 # Rated Loass Ratio R
        Tc_param = 4.0 # Oil time constant Tc
        TA0 = st.slider("Initial ambient/oil temperature (default: 25°C)", min_value=0, max_value=50, value=25)
        P_param = 20.0 # Active Power (example, will be overridden by load data)
        Q_param = 0.0 # Reactive Power (example)
        V_param = 240.0 # Voltage (example, will be overridden by load data)
        S_param = pow((P_param*P_param + Q_param*Q_param), 1/2) # Apparent power
        I_param = S_param/V_param/pf if pf > 0 else 0 # Current
        deltaTTO_R_param = 50.0 # Top-oil rise over ambient at rated load
        deltaTH_R_param = 80.0-deltaTTO_R_param # Winding hottest-spot rise over top-oil at rated load
        Tcw_param = 0.5 # Winding time constant
        kVA_param = 25.0 # Transformer kVA rating (example, will be overridden by actual data)
        Vpu_param = V_param/240.0 # Voltage per unit
        Ipu_param = I_param/kVA_param*240.0 if kVA_param > 0 else 0 # Current per unit
    
        n_param = 0.9 # Oil exponent
        m_param = 0.8 # Winding exponent
    
        N_Runs = st.slider("Number of Monte Carlo simulations (default: 10)", min_value=1, max_value=200, value=10)
    
        # Parameter ranges for Monte Carlo simulation
        st.markdown("**Monte Carlo Simulation Parameter Ranges:**")
        col1_params, col2_params = st.columns(2)
        with col1_params:
            dT_OR_min = float(st.text_input("Top oil rise min (°C, default: 31.0)", value="31.0"))
            dT_HR_min = float(st.text_input("Hot spot rise min (°C, default: 15.0)", value="15.0"))
            R_C2I_min = float(st.text_input("Loss ratio (R) min (default: 2.4)", value="2.4"))
            tau_R_min = float(st.text_input("Oil time const (Tc) min (hrs, default: 4.9)", value="4.9"))
            tau_W_min = float(st.text_input("Winding time const (Tcw) min (hrs, default: 0.5)", value="0.5"))
        with col2_params:
            dT_OR_max = float(st.text_input("Top oil rise max (°C, default: 65.0)", value="65.0"))
            dT_HR_max = float(st.text_input("Hot spot rise max (°C, default: 20.0)", value="20.0"))
            R_C2I_max = float(st.text_input("Loss ratio (R) max (default: 3.6)", value="3.6"))
            tau_R_max = float(st.text_input("Oil time const (Tc) max (hrs, default: 14.0)", value="14.0"))
            tau_W_max = float(st.text_input("Winding time const (Tcw) max (hrs, default: 0.6)", value="0.6"))
            
        # Retrieve data from session state
        meter_reads_df = st.session_state.meter_reads_df
        meter_info_df = st.session_state.meter_info_df
        ambient_temp_df = st.session_state.ambient_temperature_df
        
        # Determine unique transformers to analyze
        unique_meters = meter_reads_df['meter_id'].unique()
        transformers_to_analyze = []
        for meter in unique_meters:
            transformer_series = get_transformer(meter, meter_info_df) # Utility function
            if not transformer_series.empty:
                transformers_to_analyze.append(transformer_series.values[0])
        transformers_to_analyze = sorted(list(set(transformers_to_analyze)))
        all_transformers = transformers_to_analyze
        
        transformer_selection = st.selectbox("**Select transformers to analyze:**", ["Select all", "Deselect all"])
        
        if transformer_selection == "Select all":
            transformers_to_analyze = st.multiselect(
                label="Selected transformers:",
                options=all_transformers,
                default=all_transformers)
        elif transformer_selection == "Deselect all":
            transformers_to_analyze = st.multiselect(
                label="Selected transformers:",
                options=all_transformers,
                default=[])
        
        # Button to trigger the analysis
        if st.button("Run Thermal Analysis"):
            
            st.session_state.dt = dt
            
            st.session_state.thermal_analysis_completed = False # Reset completion flag
            st.session_state.all_results_cache = {} # Clear previous results
            st.info("Starting thermal analysis...")
    
            # Progress bars for visual feedback
            progress_col1, progress_col2 = st.columns([1, 4])
            with progress_col1:
                st.markdown("**Overall Progress:**")
                transformer_progress_bar = st.progress(0)
            with progress_col2:
                st.markdown("**Current Transformer Simulation Progress:**")
                simulation_progress_bar = st.progress(0)
            status_text = st.empty() # Placeholder for status messages
            st.markdown("---")
    
            total_transformers = len(transformers_to_analyze)
            results_display_area = st.container() # Container to display results incrementally
    
            # Loop through each transformer and perform analysis
            for i, transformer_id in enumerate(transformers_to_analyze):
                st.session_state.current_transformer = transformer_id # Update current transformer in session state
                # Update overall progress bar
                transformer_progress_value = int(((i + 1) / total_transformers) * 100) if total_transformers > 0 else 0
                transformer_progress_bar.progress(transformer_progress_value)
                status_text.text(
                    f"Processing Transformer {i+1}/{total_transformers}: {transformer_id}\n"
                    f"Running {N_Runs} simulations..."
                )
    
                try:
                    # Perform the thermal analysis using the algorithm
                    top_oil_results, hot_spot_results, ambient_temperature_processed, \
                    df_transformer_processed, time_processed, Faa_results, yearly_loss, \
                    sim_result = doThermalAnalysis(
                        transformer=transformer_id,
                        ambient_temperature=ambient_temp_df.copy(), # Pass a copy to avoid modification
                        meter_consumer_transformer=meter_info_df,
                        meter_reads=meter_reads_df,
                        N_Runs=N_Runs, pf=pf, dt=dt, R=R_param, Tc=Tc_param, TA0=TA0, P=P_param, Q=Q_param,
                        V=V_param, S=S_param, I=I_param, deltaTTO_R=deltaTTO_R_param, deltaTH_R=deltaTH_R_param,
                        Tcw=Tcw_param, kVA=kVA_param, Vpu=Vpu_param, Ipu=Ipu_param, n=n_param, m=m_param,
                        dT_OR_min=dT_OR_min, dT_OR_max=dT_OR_max, dT_HR_min=dT_HR_min, dT_HR_max=dT_HR_max,
                        R_C2I_min=R_C2I_min, R_C2I_max=R_C2I_max, tau_R_min=tau_R_min, tau_R_max=tau_R_max,
                        tau_W_min=tau_W_min, tau_W_max=tau_W_max,
                        simulation_progress_bar=simulation_progress_bar, # Pass progress bar for updates
                        status_text=status_text, # Pass status text for updates
                        transformer_idx=i+1,
                        total_transformers=total_transformers
                    )
    
                    # Store results for the current transformer
                    current_transformer_results = {
                        'top_oil': top_oil_results, 'hot_spot': hot_spot_results,
                        'ambient_temperature': ambient_temperature_processed,
                        'transformer_data': df_transformer_processed,
                        'time': time_processed, 'Faa': Faa_results, 'yearly_loss': yearly_loss,
                        'metrics': { # Calculate summary metrics
                            'max_hourly_aging': Faa_results.max().max() if not Faa_results.empty else 0,
                            'yearly_aging_hours': max(yearly_loss.values()) if yearly_loss else 0,
                            'max_hot_spot': hot_spot_results.max().max() if not hot_spot_results.empty else 0,
                            'max_top_oil': top_oil_results.max().max() if not top_oil_results.empty else 0
                        },
                        'params': sim_result
                    }
                    st.session_state.all_results_cache[transformer_id] = current_transformer_results
                                  
                    # --- Display results for the current transformer immediately ---
                    with results_display_area: # Use the pre-defined container
                        display_thermal_results(dt, meter_info_df, current_transformer_results, transformer_id)
                        
                except Exception as e:
                    st.error(f"Failed to process transformer {transformer_id}: {e}")
                    import traceback
                    st.error(traceback.format_exc()) # Print full traceback for debugging
                    continue # Move to the next transformer
    
            # After loop completion
            if st.session_state.all_results_cache:
                st.session_state.thermal_analysis_completed = True # Mark analysis as completed if results exist
    
            transformer_progress_bar.progress(100) # Set overall progress to 100%
            simulation_progress_bar.progress(100) # Set simulation progress to 100%
    
            if st.session_state.thermal_analysis_completed:
                status_text.success("All selected transformers processed.")
                
                # Display summary metrics
                st.markdown("---")
                st.header("Overall Analysis Summary")
                
                all_results = st.session_state.all_results_cache
                meter_info_df_summary = st.session_state.meter_info_df # Use for EV count
        
                if all_results: # Check if there are any results to summarize
                    processed_transformer_ids = list(all_results.keys())
                    total_evs = sum(get_transformer_ev_count(t, meter_info_df_summary) for t in processed_transformer_ids) # Utility function
        
                    # Calculate overall peak hot spot temperature
                    all_hot_spot_maxes = [
                        r['metrics']['max_hot_spot'] for r in all_results.values()
                        if r['metrics']['max_hot_spot'] is not None and pd.notna(r['metrics']['max_hot_spot'])
                    ]
                    max_temp_overall = max(all_hot_spot_maxes) if all_hot_spot_maxes else 0.0
        
                    # Display overall metrics
                    col1_sum, col2_sum, col3_sum = st.columns(3)
                    with col1_sum:
                        st.metric("Total Transformers Processed", len(all_results))
                    with col2_sum:
                        st.metric("Total EVs in System (for processed transformers)", total_evs)
                    with col3_sum:
                        st.metric("Peak Hot Spot Temp (Overall)", f"{max_temp_overall:.1f}°C")
        
                    # Download button for summary metrics of all processed transformers
                    st.download_button(
                        label="Download Overall Summary Metrics (CSV)",
                        data=convert_results_to_csv(all_results).encode('utf-8'), # Utility function
                        file_name="all_transformers_metrics_summary.csv",
                        mime="text/csv",
                        key="download_summary_csv_overall"
                    )
            else:
                status_text.warning("Analysis run, but no transformer data was successfully processed or an error occurred.")
    
        elif st.session_state.thermal_analysis_completed and not st.session_state.all_results_cache:
            # This case might occur if the run button was clicked but all transformers failed
            st.warning("Analysis run attempted, but no results were generated. Please check inputs or error messages above.")
        elif not st.session_state.thermal_analysis_completed and action == "Run Thermal Analysis":
            # Prompt user to run analysis if parameters are set but button not clicked
             st.info("Configure parameters and click 'Run Thermal Analysis' to generate and view results.")

if action == "Raw Data Analysis":
    # May need to save results with session state
    raw_data_analysis_choice = st.selectbox("What would you like to do with the data?", ["Plot transformer loading profiles"])
    
    # Retrieve data from session state
    meter_reads_df = st.session_state.meter_reads_df
    meter_info_df = st.session_state.meter_info_df
    
    if raw_data_analysis_choice == "Plot transformer loading profiles":
        
        unique_meters = meter_reads_df['meter_id'].unique()
        transformers_to_analyze = []
        for meter in unique_meters:
            transformer_series = get_transformer(meter, meter_info_df) # Utility function
            if not transformer_series.empty:
                transformers_to_analyze.append(transformer_series.values[0])
        transformers_to_analyze = sorted(list(set(transformers_to_analyze)))
        all_transformers = transformers_to_analyze
        
        transformer_selection = st.selectbox("**Select transformers to analyze:**", ["Select all", "Deselect all"])
        
        if transformer_selection == "Select all":
            transformers_to_analyze = st.multiselect(
                label="Selected transformers:",
                options=all_transformers,
                default=all_transformers)
        elif transformer_selection == "Deselect all":
            transformers_to_analyze = st.multiselect(
                label="Selected transformers:",
                options=all_transformers,
                default=[])
        
        if st.button("Plot transformer loading profiles"):
            for transformer_id_val in transformers_to_analyze: # Renamed variable to avoid conflict
                try:
                    with st.expander(f"View Loading Profile for {transformer_id_val}", expanded=True):
                        # transparency = 0.5 # This variable was defined but not used
                        print(f"Generating graph for {transformer_id_val}...")
                        
                        # Corrected aggregation logic:
                        meters_under_transformer_list = get_meters(transformer_id_val, meter_info_df) # Use the function
                        
                        if not meters_under_transformer_list:
                            print(f"No meters found for transformer {transformer_id_val}. Skipping.")
                            continue
        
                        kwh_sum_hourly = pd.Series(dtype=float) # Initialize an empty Series
                        
                        is_first_meter = True
                        for meter_id_loopvar in meters_under_transformer_list: # Renamed loop variable
                            # Extract the series for the current meter
                            # Ensure that meter_reads is indexed by DATATIMESTAMP for selection and resampling
                            meter_kwh_series = meter_reads[meter_reads['meter_id'] == meter_id_loopvar]['kwh_per_interval']
                            
                            if meter_kwh_series.empty:
                                print(f"No 'kwh_per_interval' data for meter_id {meter_id_loopvar} under transformer {transformer_id_val}. Skipping this meter.")
                                continue
        
                            # Resample this meter's data to hourly sum
                            meter_kwh_hourly = meter_kwh_series.resample('h').sum()
                            
                            if is_first_meter:
                                kwh_sum_hourly = meter_kwh_hourly
                                is_first_meter = False
                            else:
                                kwh_sum_hourly = kwh_sum_hourly.add(meter_kwh_hourly, fill_value=0)
                        
                        kwh = kwh_sum_hourly # Use the final aggregated hourly sum
        
                        if kwh.empty or kwh.isnull().all(): # Check if the series is empty or all NaNs after aggregation
                            print(f"Aggregated data is empty or all NaN for transformer {transformer_id_val}. Skipping.")
                            continue
                            
                        data = pd.DataFrame()
                        data['hour'] = kwh.index.hour
                        data['kw'] = kwh.to_numpy()
        
                        # Remove rows where kw might be NaN after aggregation if some hours had no data
                        data.dropna(subset=['kw'], inplace=True)
                        if data.empty:
                            print(f"No valid data points after removing NaNs for transformer {transformer_id_val}. Skipping.")
                            continue
        
        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        cmap = sns.color_palette("crest", as_cmap=True)
                        sns.histplot(x=data['hour'], y=data['kw'], log_scale=False, cmap=cmap, ax=ax) # Pass ax to histplot
        
                        ax.set_facecolor('white')
                        sns.despine(ax=ax) # Pass ax to despine
        
                        # Set xlim robustly, ensuring data['hour'] is not empty
                        if not data['hour'].empty:
                            xlim = [data['hour'].min(), data['hour'].max()]
                            ax.set_xlim(xlim)
                        else: # Default xlim if no hour data (should be caught by earlier checks)
                            ax.set_xlim([0, 23])
        
        
                        # Set ylim robustly
                        if not data['kw'].empty:
                            ylim_max = data['kw'].max() if pd.notna(data['kw'].max()) else 1 # Handle potential all-NaN case if not caught
                            ylim = [0, max(1, ylim_max * 1.05)]
                            ax.set_ylim(ylim)
                        else: # Default ylim if no kw data
                            ax.set_ylim([0,1])
        
        
                        ax.grid()
                        
                        transformer_kVA = get_kva(transformer_id_val, meter_info_df) # Use the function
                        
                        if not data.empty: # Ensure data is not empty before groupby and quantile
                            hourly_90th = data.groupby('hour')['kw'].quantile(0.9)
                            ax.step(hourly_90th.index, hourly_90th.values,    
                                     color='red', linestyle='-', linewidth=2,    
                                     label='90th percentile')
                            
                            hourly_50th = data.groupby('hour')['kw'].quantile(0.5)
                            ax.step(hourly_50th.index, hourly_50th.values,    
                                     color='black', linestyle='-', linewidth=2,    
                                     label='50th percentile')
                            
                            hourly_10th = data.groupby('hour')['kw'].quantile(0.1)
                            ax.step(hourly_10th.index, hourly_10th.values,    
                                     color='blue', linestyle='-', linewidth=2,    
                                     label='10th percentile')
                        
                        if transformer_kVA > 0 : # Only plot if kVA is known and positive
                            ax.axhline(y=transformer_kVA, color='orange', linestyle='-', linewidth=2,
                                       label=f'Transformer capacity ({transformer_kVA} kVA)')
                        
                        current_num_meters = get_num_meters(transformer_id_val, meter_info_df) # Use the function
                        current_num_evs = get_num_evs(transformer_id_val, meter_info_df) # Use the function
                        
                        ax.set_title(f"{transformer_id_val} ({transformer_kVA} kVA, {current_num_meters} meters, {current_num_evs} EVs)")
                        
                        ax.set_xlabel("Hour of day", fontsize=8)
                        ax.set_ylabel("kW", fontsize=8)
                        ax.legend() # Add legend to show labels for percentile and capacity lines
                                            
                        pdf_buffer = generate_pdf_report(fig, transformer_id_val) # Utility function
                        base64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)
                        
                        print(f"Graph for {transformer_id_val} generated")
                except Exception as e: # Catch more specific exceptions if possible
                    print(f"Error with {transformer_id_val}. No graph generated. Error: {e}")
                    # If a figure was created before the error, ensure it's closed
                    if 'fig' in locals() and plt.fignum_exists(fig.number):
                         plt.close(fig)
                    continue
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
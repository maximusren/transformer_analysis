import streamlit as st

from utils import *
from algorithm import *

# Enable wide layout
st.set_page_config(layout="wide")

st.title('My First Streamlit App')
st.write('Welcome to my Streamlit app!')

#%% Uploading files

# Upload meter reads file
meter_reads_file_path = st.text_input("Enter path to meter data file (.csv)")

if meter_reads_file_path:
    try:
        if meter_reads_file_path.endswith('.csv'):
            meter_reads = pd.read_csv(meter_reads_file_path)
            
            col1, col2, col3 = st.columns(3)
            
            # Column 1: Raw meter reads
            with col1:
                st.subheader("Raw Meter Reads")
                st.dataframe(meter_reads)
            
            # Column 2: Mapping columns
            with col2:
                st.subheader("Map Your Columns")
                meter_id_col = st.selectbox("Meter ID", meter_reads.columns)
                timestamp_col = st.selectbox("Timestamp", meter_reads.columns)
                kwh_col = st.selectbox("kWh per Interval", meter_reads.columns)
                voltage_col = st.selectbox("Voltage", meter_reads.columns)
            
            # Column 3: Mapped data
            with col3:
                st.subheader("Mapped Meter Reads")
                meter_reads = pd.DataFrame({
                      'meter_id': meter_reads[meter_id_col],
                      'DATATIMESTAMP': meter_reads[timestamp_col],
                      'kwh_per_interval': meter_reads[kwh_col],
                      'voltage': meter_reads[voltage_col],
                  })
      
                st.dataframe(meter_reads)
            
            # Make index datetime
            meter_reads['time'] = pd.to_datetime(meter_reads['DATATIMESTAMP'])
            meter_reads.set_index('time', inplace=True)    
            
        else:
            st.error("Unsupported file type.")
            meter_reads = None
            
    except Exception as e:
        st.error(f"Error reading file: {e}")        
        
        
# Upload other meter information (meter consumer transformer mapping, kva rating of transformer, meter read multiplier, ev status)
meter_consumer_transformer_file_path = st.text_input("Enter path to other meter information (meter, consumer, and transformer mapping; kVA rating of transformer; meter read multiplier, kwown EV) (.csv)")

if meter_consumer_transformer_file_path:
    try:
        if meter_consumer_transformer_file_path.endswith('.csv'):
            meter_consumer_transformer = pd.read_csv(rf'{meter_consumer_transformer_file_path}')
            
            col1, col2, col3 = st.columns(3)
            
            # Column 1: Raw mapping data
            with col1:
                st.subheader("Raw Mapping Data")
                st.dataframe(meter_consumer_transformer)
            
            # Column 2: Mapping columns
            with col2:
                st.subheader("Map Your Columns")
    
                cols = meter_consumer_transformer.columns.tolist()
    
                meter_id_col = st.selectbox("Meter ID", cols)
                consumer_id_col = st.selectbox("Consumer ID", cols)
                transformer_id_col = st.selectbox("Transformer ID", cols)
                transformer_type_col = st.selectbox("Transformer Type", cols)
                multiplier_col = st.selectbox("Meter Read Multiplier", cols)
                known_ev_col = st.selectbox("Known EV", cols)
            
            # Column 3: Mapped data
            with col3:
                st.subheader("Mapped Data")
                meter_consumer_transformer = pd.DataFrame({
                    'meter_id': meter_consumer_transformer[meter_id_col],
                    'consumer_id': meter_consumer_transformer[consumer_id_col],
                    'transformer_id': meter_consumer_transformer[transformer_id_col],
                    'transformer_type': meter_consumer_transformer[transformer_type_col],
                    'meter_read_multiplier': meter_consumer_transformer[multiplier_col],
                    'known_ev': meter_consumer_transformer[known_ev_col],
                })
                st.dataframe(meter_consumer_transformer)
                
            # Create additional column containing kva rating
            meter_consumer_transformer['kva_rating'] = meter_consumer_transformer['transformer_type'].apply(extract_kva)
            
        else:
            st.error("Unsupported file type.")
            meter_consumer_transformer = None  
            
    except Exception as e:
        st.error(f"Error reading file: {e}")
        
# Upload ambient temperature information
ambient_temperature_file_path = st.text_input("Enter path to ambient temperature data (.csv or .pkl)")

if ambient_temperature_file_path:
    try:
        if ambient_temperature_file_path.endswith('.csv'):
            ambient_temperature = pd.read_csv(rf'{ambient_temperature_file_path}')
            
        elif ambient_temperature_file_path.endswith('.pkl'):
            ambient_temperature = pd.read_pickle(rf'{ambient_temperature_file_path}')
            
            # Convert to dataframe
            if isinstance(ambient_temperature, pd.Series):
                ambient_temperature = ambient_temperature.to_frame().reset_index()
        else:
            st.error("Unsupported file type.")
            ambient_temperature = None
            
        col1, col2, col3 = st.columns(3)
        
        # Column 1: Raw ambient temperature data
        with col1:
            st.subheader("Raw Ambient Temperature Data")
            st.dataframe(ambient_temperature)
        
        # Column 2: Mapping columns
        with col2:
            st.subheader("Map Your Columns")

            cols = ambient_temperature.columns.tolist()

            datetime_col = st.selectbox("Timestamp", cols)
            Tamb_col = st.selectbox("Ambient Temperature (C)", cols)
          
        # Column 3: Mapped ambient temperature data
        with col3:
            st.subheader("Mapped Ambient Temperature Data")
            ambient_temperature = pd.DataFrame({
                'datetime': ambient_temperature[datetime_col],
                'Tamb (C)': ambient_temperature[Tamb_col],
            })
            st.dataframe(ambient_temperature)      
            
    except Exception as e:
        st.error(f"Error reading file: {e}")
        
#%% Thermal Analysis

# Make sure the files are uploaded required for thermal analysis
if meter_reads is not None and meter_consumer_transformer is not None and ambient_temperature is not None:
    st.markdown("---")
    st.subheader("What would you like to do with the data?")
    
    # Select action
    action = st.selectbox("Choose an action", ["Select Action", "Run Thermal Analysis"])
    
    # Run thermal analysis
    if action == "Run Thermal Analysis":
        st.subheader("Thermal Analysis Settings")

        # Initial parameters
        pf = st.slider("Power factor (default: 1)", min_value=0, max_value=1, value=1)
        dt = 0.1 # Time step
        R = 5.5 # Rated power
        Tc = 4 # Time constant
        TA0 = st.slider("Initial ambient temperature (default: 25)", min_value=0, max_value=50, value=25) # Initial ambient temperature
        P = 20 # Power
        Q = 0 # Reactive power
        V = 240 # Voltage
        S = pow((P*P + Q*Q), 1/2)
        I = S/V/pf # Current 
        deltaTTO_R = 50 # Top-oil rise over ambient temperature
        deltaTH_R = 80-deltaTTO_R # Winding hottest-spot rise over top-oil temperature
        Tcw = 0.5 # Winding time constant
        kVA = 25 
        Vpu = V/240 # Voltage per unit
        Ipu = I/kVA*240 # Current per unit       
        
        # Other parameters with no range
        n = 0.9
        m = 0.8
        
        time = [] # Store times of each temperature calculation
        TO_temp = [] # Store top-oil temperature
        TH_temp = [] # Store hot-spot temperature
        Faa_list = [] # Store Faa values
        TTO = TA0 # Initial top-oil temperature
        TH = TA0 # Initial hot-spot temperature
        
        # Other parameters with ranges
        N_Runs = st.slider("Number of simulations (default: 10)", min_value=5, max_value=200, value=10) # Number of simulations
        
        col1, col2 = st.columns(2)
        
        with col1:
            dT_OR_min = float(st.text_input("Top oil rise min (default: 31.0)", value="31.0"))
            dT_HR_min = float(st.text_input("Hot spot rise min (default: 15.0)", value="15.0"))
            R_C2I_min = float(st.text_input("Loss ratio min (default: 2.4)", value="2.4"))
            tau_R_min = float(st.text_input("Oil time const min (default: 4.9)", value="4.9"))
            tau_W_min = float(st.text_input("Winding time const min (default: 0.5)", value="0.5"))
        
        with col2:
            dT_OR_max = float(st.text_input("Top oil rise max (default: 65.0)", value="65.0"))
            dT_HR_max = float(st.text_input("Hot spot rise max (default: 20.0)", value="20.0"))
            R_C2I_max = float(st.text_input("Loss ratio max (default: 3.6)", value="3.6"))
            tau_R_max = float(st.text_input("Oil time const max (default: 14.0)", value="14.0"))
            tau_W_max = float(st.text_input("Winding time const max (default: 0.6)", value="0.6"))
           
        # After the run button is pressed
        if st.button("Run Thermal Analysis"):
            st.write("Running thermal analysis with the parameters provided...")
            
            # To track progress 
            progress_col1, progress_col2 = st.columns([1, 4])
            
            with progress_col1:
                st.markdown("**Overall Progress:**")
                transformer_progress_bar = st.progress(0)
                
            with progress_col2:
                st.markdown("**Current Transformer:**")
                simulation_progress_bar = st.progress(0)
            
            status_text = st.empty()  # For detailed status messages
            
            # Add some spacing
            st.markdown("---")
            
            # Thermal analysis
            all_results = {}
            
            # Get list of transformers from meter reads (not meter consumer transformer)
            unique_meters = meter_reads['meter_id'].unique()
            
            # Get corresponding transformers for these meters
            transformers = []
            for meter in unique_meters:
                transformer = get_transformer(meter, meter_consumer_transformer)
                if not transformer.empty:
                    transformers.append(transformer.values[0])
            
            # Remove duplicates and keep order
            transformers = sorted(list(set(transformers)))
            
            # Temp code for testing
            transformers = ['T3913']
            # transformers = transformers[0:5]
            
            total_transformers = len(transformers)
            
            for i, transformer in enumerate(transformers):
                # Update transformer-level progress
                transformer_progress = int((i / total_transformers) * 100)
                transformer_progress_bar.progress(transformer_progress)
                
                # Initial status for this transformer
                status_text.text(
                    f"**Processing Transformer {i+1}/{total_transformers}:** {transformer}\n"
                    f"Preparing to run {N_Runs} simulations..."
                )
                
                try:
                    # Run thermal analysis
                    top_oil_results, hot_spot_results, ambient_temp, df_transformer, time, Faa_results, yearly_loss = doThermalAnalysis(
                        transformer, 
                        ambient_temperature, 
                        meter_consumer_transformer, 
                        meter_reads, 
                        N_Runs, 
                        pf, 
                        dt, 
                        R, 
                        Tc, 
                        TA0, 
                        P, 
                        Q, 
                        V, 
                        S, 
                        I, 
                        deltaTTO_R, 
                        deltaTH_R, 
                        Tcw, 
                        kVA,
                        Vpu, 
                        Ipu, 
                        n, 
                        m,
                        float(dT_OR_min), 
                        float(dT_OR_max), 
                        float(dT_HR_min), 
                        float(dT_HR_max), 
                        float(R_C2I_min), 
                        float(R_C2I_max), 
                        float(tau_R_min), 
                        float(tau_R_max), 
                        float(tau_W_min), 
                        float(tau_W_max),
                        simulation_progress_bar,
                        status_text,
                        transformer_idx=i+1,
                        total_transformers=total_transformers
                    )
                    
                    # Store results
                    all_results[transformer] = {
                        'top_oil': top_oil_results,
                        'hot_spot': hot_spot_results,
                        'ambient_temp': ambient_temp,
                        'transformer_data': df_transformer,
                        'time': time,
                        'Faa': Faa_results,
                        'yearly_loss': yearly_loss,
                        
                        
                        'metrics': {
                            # Most loss of life in one hour (max Faa value across all runs)
                            'max_hourly_aging': Faa_results.max().max(),
                            
                            # Yearly aging (already being calculated in yearly_loss)
                            'yearly_aging_hours': max(yearly_loss.values()),
                            
                            # Max hot spot temp across all runs
                            'max_hot_spot': hot_spot_results.max().max(),
                            
                            # Max top oil temp across all runs
                            'max_top_oil': top_oil_results.max().max()
                        }
                    }
                    
                    # Plot results for this transformer
                    fig = plot_results2(
                        transformer,
                        top_oil_results,
                        hot_spot_results,
                        ambient_temp,
                        df_transformer,
                        time,
                        ambient_temp['Tamb (C)'],
                        df_transformer['kwh_per_interval'],
                        dt,
                        meter_consumer_transformer
                    )
                    st.pyplot(fig)
                    
                    # Show summary stats
                    with st.expander(f"Transformer {transformer} - Key Metrics"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Worst Hourly Aging", 
                                     f"{all_results[transformer]['metrics']['max_hourly_aging']:.2f}",
                                     help="Maximum loss of life in any single hour")
                            
                            st.metric("Yearly Aging Impact", 
                                     f"{all_results[transformer]['metrics']['yearly_aging_hours']:.1f} hours",
                                     help="Equivalent hours of aging per year")
                        
                        with col2:
                            st.metric("Peak Hot Spot", 
                                     f"{all_results[transformer]['metrics']['max_hot_spot']:.1f}°C",
                                     help="Maximum winding hot spot temperature")
                            
                            st.metric("Peak Top Oil", 
                                     f"{all_results[transformer]['metrics']['max_top_oil']:.1f}°C",
                                     help="Maximum top oil temperature")
                    
                except Exception as e:
                    st.error(f"Failed to process transformer {transformer}: {str(e)}")
                    continue
            
            # Update progress bars to 100%
            transformer_progress_bar.progress(100)
            simulation_progress_bar.progress(100)
            
            # Show completion message
            status_text.success("All transformers processed successfully.")
            
            # Summary statistics
            st.subheader("Analysis Summary")
            if all_results:
                total_evs = sum(get_transformer_ev_count(t, meter_consumer_transformer) for t in transformers)
                max_temp = max(r['hot_spot'].max().max() for r in all_results.values())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Transformers Processed", len(all_results))
                with col2:
                    st.metric("Total EVs in System", total_evs)
                with col3:
                    st.metric("Peak Hot Spot Temp", f"{max_temp:.1f}°C")
                
                # Option to download results
                st.download_button(
                    label="Download Summary Metrics as CSV",
                    data=convert_results_to_csv(all_results),
                    file_name="transformer_metrics_summary.csv",
                    mime="text/csv"
                )
                
            else:
                st.warning("No transformers were successfully processed")
                            
        









































import streamlit as st
import base64
import io
from matplotlib.backends.backend_pdf import PdfPages
from utils import *
from algorithm import *

#%% Initialize all session state variables
if 'meter_reads_df' not in st.session_state:
    st.session_state.meter_reads_df = None
if 'meter_info_df' not in st.session_state:
    st.session_state.meter_info_df = None
if 'ambient_temperature_df' not in st.session_state:
    st.session_state.ambient_temperature_df = None

if 'files_uploaded' not in st.session_state:
    st.session_state.files_uploaded = {
        'meter_reads': False,
        'meter_information': False,
        'ambient_temperature_data': False
    }
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'current_transformer' not in st.session_state:
    st.session_state.current_transformer = None
# if 'download_requested' not in st.session_state:
#     st.session_state.download_requested = None   # For download confirmations
    
#%%
# Enable wide layout
st.set_page_config(layout="wide")

st.title('My First Streamlit App')
st.write('Welcome to my Streamlit app!')

#%% Uploading files

uploaded_meter_reads_file = st.file_uploader(
    "Upload Meter Data File (.csv)",
    type=['csv'],
    key="meter_reads_uploader",
    help="CSV file containing Meter ID, Timestamp, kWh per Interval, Voltage"
)

if uploaded_meter_reads_file is not None:
    try:
        # Read directly from the uploaded file object
        meter_reads_raw = pd.read_csv(uploaded_meter_reads_file)
        st.success("Meter Reads file uploaded successfully!")

        col1, col2, col3 = st.columns(3)

        # Column 1: Raw meter reads preview
        with col1:
            st.subheader("Raw Meter Reads (Preview)")
            st.dataframe(meter_reads_raw.head())

        # Column 2: Mapping columns
        with col2:
            st.subheader("Map Your Columns")
            available_cols = meter_reads_raw.columns.tolist()
            meter_id_col = st.selectbox("Meter ID", available_cols, key="map_meter_id")
            timestamp_col = st.selectbox("Timestamp", available_cols, key="map_timestamp")
            kwh_col = st.selectbox("kWh per Interval", available_cols, key="map_kwh")
            voltage_col = st.selectbox("Voltage", available_cols, key="map_voltage")

        # Column 3: Mapped data preview
        with col3:
            st.subheader("Mapped Meter Reads (Preview)")
            # Create the mapped DataFrame
            meter_reads = pd.DataFrame({
                  'meter_id': meter_reads_raw[meter_id_col],
                  'DATATIMESTAMP': meter_reads_raw[timestamp_col],
                  'kwh_per_interval': meter_reads_raw[kwh_col],
                  'voltage': meter_reads_raw[voltage_col],
              })
            st.dataframe(meter_reads.head())

        # Process and store in session state 
        meter_reads['time'] = pd.to_datetime(meter_reads['DATATIMESTAMP'], format='%Y-%m-%d %H:%M:%S')
        meter_reads.set_index('time', inplace=True)
        st.session_state.meter_reads_df = meter_reads # Store processed df
        st.session_state.files_uploaded['meter_reads'] = True

    except Exception as e:
        st.error(f"Error processing Meter Reads file: {e}")
        st.session_state.meter_reads_df = None # Reset on error
        st.session_state.files_uploaded['meter_reads'] = False
else:
    # Reset if file is removed
    st.session_state.files_uploaded['meter_reads'] = False if st.session_state.meter_reads_df is None else True


# Meter information upload
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

        # Column 1: Raw mapping data preview
        with col1:
            st.subheader("Raw Mapping Data (Preview)")
            st.dataframe(meter_info_raw.head())

        # Column 2: Mapping columns
        with col2:
            st.subheader("Map Your Columns")
            available_cols = meter_info_raw.columns.tolist()
            meter_id_col = st.selectbox("Meter ID", available_cols, key="map_info_meter_id")
            consumer_id_col = st.selectbox("Consumer ID", available_cols, key="map_info_consumer_id")
            transformer_id_col = st.selectbox("Transformer ID", available_cols, key="map_info_transformer_id")
            transformer_type_col = st.selectbox("Transformer Type", available_cols, key="map_info_transformer_type")
            multiplier_col = st.selectbox("Meter Read Multiplier", available_cols, key="map_info_multiplier")
            known_ev_col = st.selectbox("Known EV", available_cols, key="map_info_known_ev")

        # Column 3: Mapped data preview
        with col3:
            st.subheader("Mapped Data (Preview)")
            meter_consumer_transformer = pd.DataFrame({
                'meter_id': meter_info_raw[meter_id_col],
                'consumer_id': meter_info_raw[consumer_id_col],
                'transformer_id': meter_info_raw[transformer_id_col],
                'transformer_type': meter_info_raw[transformer_type_col],
                'meter_read_multiplier': meter_info_raw[multiplier_col],
                'known_ev': meter_info_raw[known_ev_col],
            })
            st.dataframe(meter_consumer_transformer.head())

        # Process and store in session state
        meter_consumer_transformer['kva_rating'] = meter_consumer_transformer['transformer_type'].apply(extract_kva)
        st.session_state.meter_info_df = meter_consumer_transformer
        st.session_state.files_uploaded['meter_information'] = True

    except Exception as e:
        st.error(f"Error processing Meter Information file: {e}")
        st.session_state.meter_info_df = None
        st.session_state.files_uploaded['meter_information'] = False
else:
    # Reset if file is removed
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
        # Determine file type from name and read accordingly
        file_name = uploaded_ambient_temperature_file.name
        
        if file_name.lower().endswith('.csv'):
            ambient_temperature_raw = pd.read_csv(uploaded_ambient_temperature_file)
        elif file_name.lower().endswith('.pkl'):
            ambient_temperature_raw = pd.read_pickle(uploaded_ambient_temperature_file)
            # Convert Series to DataFrame 
            if isinstance(ambient_temperature_raw, pd.Series):
                ambient_temperature_raw = ambient_temperature_raw.to_frame().reset_index()
        else:
            st.error("Unsupported file type for Ambient Temperature. Please use .csv or .pkl")
            raise ValueError("Unsupported file type") # Stop processing

        st.success("Ambient Temperature file uploaded successfully!")

        col1, col2, col3 = st.columns(3)

        # Column 1: Raw ambient temperature data preview
        with col1:
            st.subheader("Raw Ambient Temperature Data (Preview)")
            st.dataframe(ambient_temperature_raw.head())

        # Column 2: Mapping columns
        with col2:
            st.subheader("Map Your Columns")
            available_cols = ambient_temperature_raw.columns.tolist()
            datetime_col = st.selectbox("Timestamp", available_cols, key="map_amb_datetime")
            tamb_col = st.selectbox("Ambient Temperature (C)", available_cols, key="map_amb_temp")

        # Column 3: Mapped ambient temperature data preview
        with col3:
            st.subheader("Mapped Ambient Temperature Data (Preview)")
            ambient_temperature = pd.DataFrame({
                'datetime': ambient_temperature_raw[datetime_col],
                'Tamb (C)': ambient_temperature_raw[tamb_col],
            })
            st.dataframe(ambient_temperature.head())

        # Process and store in session state
        st.session_state.ambient_temperature_df = ambient_temperature
        st.session_state.files_uploaded['ambient_temperature_data'] = True

    except Exception as e:
        st.error(f"Error processing Ambient Temperature file: {e}")
        st.session_state.ambient_temperature_df = None
        st.session_state.files_uploaded['ambient_temperature_data'] = False
else:
    # Reset if file is removed
    st.session_state.files_uploaded['ambient_temperature_data'] = False if st.session_state.ambient_temperature_df is None else True
        
#%% Thermal Analysis

# Make sure the files are uploaded required for thermal analysis
required_files = ['meter_reads', 'meter_information', 'ambient_temperature_data']
if not all(st.session_state.files_uploaded[file] for file in required_files):
    st.stop()

# Show main analysis UI only when files are ready
st.markdown("---")
st.subheader("What would you like to do with the data?")

# Select action
action = st.selectbox("Choose an action", ["Select Action", "Run Thermal Analysis"])

# Run thermal analysis
if action == "Run Thermal Analysis":
    st.subheader("Thermal Analysis Settings")
    st.session_state.analysis_run = True

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
        # transformers = ['T3913']
        transformers = transformers[0:3]
        
        total_transformers = len(transformers)
        
        for i, transformer in enumerate(transformers):
            # Update transformer-level progress
            st.session_state.current_transformer = transformer
            transformer_progress = int((i / total_transformers) * 100)
            transformer_progress_bar.progress(transformer_progress)
            
            # Initial status for this transformer
            status_text.text(
                f"**Processing Transformer {i+1}/{total_transformers}:** {transformer}\n"
                f"Preparing to run {N_Runs} simulations..."
            )
            
            try:
                # Run thermal analysis
                top_oil_results, hot_spot_results, ambient_temperature, df_transformer, time, Faa_results, yearly_loss, sim_result = doThermalAnalysis(
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
                    'ambient_temperature': ambient_temperature,
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
                    },
                    
                    'params': sim_result
                }
                
                # Plot results for this transformer
                fig = plot_results2(
                    transformer,
                    top_oil_results,
                    hot_spot_results,
                    df_transformer,
                    time,
                    ambient_temperature['Tamb (C)'],
                    df_transformer['kwh_per_interval'],
                    dt,
                    meter_consumer_transformer
                )
                  
                pdf_buffer = generate_pdf_report(fig, transformer)
                base64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
                
                st.markdown(f"### {transformer} Thermal Analysis Results")

                # Preview the PDF in a smaller viewer (BEFORE key metrics)
                with st.expander(f"{transformer} Thermal Analysis PDF"):
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="450" height="600" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                                    
                # Show summary stats
                with st.expander(f"{transformer} Summary Metrics"):
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
                
                # Simulation Parameters Table
                with st.expander("Simulation Parameters Used"):
                    sim_df = pd.DataFrame(all_results[transformer]['params'])
                    st.dataframe(sim_df.style.format(precision=2))
                    
                sim_csv = sim_df.to_csv(index=False)
                st.download_button(
                    label="Download simulation parameters (CSV)",
                    data=sim_csv,
                    file_name=f"{transformer}_simulation_parameters.csv",
                    mime='text/csv'
                )

                combined_data = create_download_data(transformer, all_results[transformer])
                csv = combined_data.to_csv(index=True)
                st.download_button(
                    label=f"Download all results as CSV for {transformer} (Warning: large file size)",
                    data=csv,
                    file_name=f"{transformer}_thermal_analysis_results.csv",
                    mime='text/csv'
                )
                                    
                        
                        
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
                file_name="all_transformer_metrics_summary.csv",
                mime="text/csv"
            )
            
        else:
            st.warning("No transformers were successfully processed")
                        
        






































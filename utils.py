import os
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import pickle
import re
import matplotlib.dates as mdates

# Get transformer kVA
def extract_kva(transformer_type):
    # Use regular expression to find number before 'kVA'
    match = re.search(r'(\d+)\s*kVA', transformer_type)
    if match:
        return int(match.group(1))
    else:
        return None 
    
def get_kva(transformer, meter_consumer_transformer): #kva of a given transformer
    return int(re.search(r'-(\d+)\s*kVA', meter_consumer_transformer[meter_consumer_transformer['transformer_id'] == transformer]['transformer_type'].to_string()).group(1))

# Array with the ids of all meters under a transformer
def get_meters(transformer, meter_consumer_transformer):
    return meter_consumer_transformer[meter_consumer_transformer['transformer_id'] == transformer]['meter_id'].tolist()

# Number of meters under a transformer
def get_num_meters(transformer, meter_consumer_transformer):
    return len(meter_consumer_transformer[meter_consumer_transformer['transformer_id'] == transformer]['meter_id'].tolist())

# Array with the ids of all consumers under a transformer
def get_consumers(transformer, meter_consumer_transformer): 
    return meter_consumer_transformer[meter_consumer_transformer['transformer_id'] == transformer]['consumer_id'].tolist()

# Number of consumers under a transformer
def get_num_consumers(transformer, meter_consumer_transformer): 
    return len(meter_consumer_transformer[meter_consumer_transformer['transformer_id'] == transformer]['consumer_id'].tolist())

def get_num_evs(transformer, meter_consumer_transformer):
    num_evs = 0
    for meter in get_meters(transformer, meter_consumer_transformer):
        num_evs = num_evs + meter_consumer_transformer[meter_consumer_transformer['meter_id'] == meter]['known_ev'].values[0]
    return num_evs

# Array of the meters connected to a consumer
def get_meters_under_consumer(consumer, meter_consumer_transformer):
    return meter_consumer_transformer[meter_consumer_transformer['consumer_id'] == consumer]['meter_id'].tolist()

# Array of the consumers with the given meter (always just one consumer)
def get_consumer_with_meter(consumer, meter_consumer_transformer): 
    return meter_consumer_transformer[meter_consumer_transformer['meter_id'] == consumer]['consumer_id'].tolist()

def get_consumer_ev_count(consumer, meter_consumer_transformer):
    return meter_consumer_transformer[meter_consumer_transformer['consumer_id'] == consumer]["known_ev"].tolist()[0]

def get_transformer_ev_count(transformer, meter_consumer_transformer):
    num_evs_under_transformer = 0
    for consumer in get_consumers(transformer, meter_consumer_transformer):
        num_evs_under_transformer += get_consumer_ev_count(consumer, meter_consumer_transformer)
    return num_evs_under_transformer  

# Get the transformer a meter belongs to
def get_transformer(meter, meter_consumer_transformer):
    return meter_consumer_transformer[meter_consumer_transformer['meter_id'] == meter]['transformer_id']

# Convert the results dictionary to a downloadable CSV format
def convert_results_to_csv(results_dict):

    # Create a DataFrame from the metrics
    metrics_data = []
    for transformer, data in results_dict.items():
        metrics = data.get('metrics', {})
        metrics_data.append({
            'Transformer': transformer,
            'Max Hourly Aging': metrics.get('max_hourly_aging', 0),
            'Yearly Aging Hours': metrics.get('yearly_aging_hours', 0),
            'Max Hot Spot (°C)': metrics.get('max_hot_spot', 0),
            'Max Top Oil (°C)': metrics.get('max_top_oil', 0)
        })
    
    df = pd.DataFrame(metrics_data)
    
    # Convert to CSV
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()
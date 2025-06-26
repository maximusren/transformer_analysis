Created by Maximus Ren
maximusren26@gmail.com

Instructions to run:
1. Install streamlit (https://docs.streamlit.io/get-started/installation)
2. Navigate to the file that contains main.py
3. Open Windows Powershell
4. In the command line type: streamlit run main.py


Uploading files:
Meter data file:
 - Purpose: smart meter reads are required for thermal analysis

 - Columns required: meter_id, Timestamp, kwh_per_interval, voltage

 - Column descriptions:

	meter_id: unique identifier of a meter
	Timestamp: date and time of the recorded data
	kwh_per_interval: accumulated kwh since the last reading. The timestamp value is the end of the interval.
	voltage: measured voltage at midnight

 - Example (5 rows):

	meter_id	timestamp		kwh_per_interval	voltage		
	M1474		2023-10-01 00:00:00	0.7019999999993161	245.5
	M1474		2023-10-01 01:00:00	0.6920000000009168	
	M1474		2023-10-01 02:00:00	0.6659999999992579	
	M1474		2023-10-01 03:00:00	0.6490000000012515	
	M1474		2023-10-01 04:00:00	0.661999999998443		


Meter information file:
 - Purpose: provide mapping between transformers and meters and consumers, transformer kVAs, meter_read_multiplier, EV status  

 - Columns required: meter id, consumer id, transformer id, transformer type, meter read multiplier, known EV

 - Column descriptions:

	meter_id: unique identifier of a meter
	consumer_id: unique identifier of a consumer (service)
	transformer_id: unique identifier of a transformer
	transformer_type: description of the transformer that includes its size in kVA
	meter_read_multiplier: multiplier value for meters that measure current with a current transformer (CT). Must be applied to kW and kWh values recorded by the meter.
	known_ev: Boolean, 1 = known EV, 0 = unknown

 - Example (5 rows): 

	meter_id	consumer_id	transformer_id	transformer_type	meter_read_multiplier	known_ev
	M5812		C253845		T100246		UG-15 kVA-1Ph		1			0
	M556		C351107		T100246		UG-15 kVA-1Ph		1			0
	M7432		C482293		T100246		UG-15 kVA-1Ph		1			0
	M3322		C153019		T102632		UG-25 kVA-1Ph		1			0
	M10566		C211394		T102632		UG-25 kVA-1Ph		1			1


Ambient temperature file:
 - Purpose: Ambient temperature is required for accurate thermal analysis

 - Columns required: time, temperature

 - Column descriptions:
	
	datetime: the time
	Tamb (C): Ambient temperature in Celsius

 - Example (5 rows):

	datetime		Tamb (C)
	2023-10-01 00:00:00	22.055555555555557
	2023-10-01 01:00:00	21.333333333333336
	2023-10-01 02:00:00	20.666666666666668
	2023-10-01 03:00:00	20.72222222222222
	2023-10-01 04:00:00	21.38888888888889

	




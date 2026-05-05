import os
import io
import urllib.request
import pandas as pd
import numpy as np
import obspy
from obspy import UTCDateTime, read_inventory
from obspy.clients.fdsn import Client
import matplotlib
matplotlib.use('Agg') # Run in background
import matplotlib.pyplot as plt
from scipy.signal import welch

# Assuming your custom PPSD module is correctly configured in your environment
from ppsd_accel.custom_ppsd import CustomPPSD as PPSD
from obspy.imaging.cm import pqlx

# ==========================================
# 1. Instrument Response Plotter
# ==========================================
def plot_instrument_response(inventory, trace_id, output_dir, output_unit):
    """Saves the Bode plot (Amplitude & Phase) of the instrument response."""
    filename = os.path.join(output_dir, f"response_{trace_id}.png")
    print(f"    -> Saving Instrument Response ({output_unit}) to {filename}...")
    inventory.plot_response(min_freq=0.001, output=output_unit, outfile=filename)

# ==========================================
# 2. Raw Waveform Plotter
# ==========================================
def plot_raw_waveform(trace, trace_id, output_dir):
    """Plots and saves the raw uncorrected waveform in Digital Counts."""
    filename = os.path.join(output_dir, f"raw_waveform_{trace_id}.png")
    print(f"    -> Saving Raw Waveform to {filename}...")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trace.times(), trace.data, color='black', linewidth=0.5)
    
    ax.set_title(f"Raw Waveform: {trace_id}")
    ax.set_ylabel("Digital Counts")
    ax.set_xlabel("Time (seconds)")
    ax.margins(x=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# 3. Corrected Waveform Plotter
# ==========================================
def plot_corrected_waveform(trace_corr, trace_id, output_dir, output_unit):
    """Plots and saves the waveform after instrument correction."""
    filename = os.path.join(output_dir, f"corrected_waveform_{trace_id}.png")
    print(f"    -> Saving Corrected Waveform to {filename}...")
    
    # Adjust labels based on sensor type
    y_label = "Acceleration (m/s²)" if output_unit == "ACC" else "Velocity (m/s)"
    title_type = "Ground Acceleration" if output_unit == "ACC" else "Ground Velocity"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trace_corr.times(), trace_corr.data, color='blue', linewidth=0.5)
    
    ax.set_title(f"Corrected Waveform ({title_type}): {trace_id}")
    ax.set_ylabel(y_label)
    ax.set_xlabel("Time (seconds)")
    ax.margins(x=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# 4. PPSD Plotter (ObsPy Probabilistic PSD)
# ==========================================
def plot_ppsd_obspy(trace, inventory, trace_id, output_dir, sensor_type):
    """Calculates and plots the Probabilistic PSD using ObsPy."""
    filename = os.path.join(output_dir, f"ppsd_obspy_{trace_id}.png")
    print(f"    -> Saving ObsPy PPSD Plot to {filename}...")

    # Only pass 'accelerometer' if that's the sensor type, otherwise None for default seismometer
    special_handling = "accelerometer" if sensor_type == "accelerometer" else None
    
    ppsd = PPSD(trace.stats, metadata=inventory, special_handling=special_handling)
    ppsd.add(trace)
    ppsd.plot(filename=filename, show=False, cmap=pqlx)

# ==========================================
# Instrument Correction Function (Logic)
# ==========================================
def correct_instrument_response(trace, inventory, output_unit):
    """Removes instrument response to convert digital counts to ACC or VEL."""
    st_single = obspy.Stream(traces=[trace.copy()])
    pre_filt = [0.05, 0.1, 40.0, 45.0] 
    
    st_single.remove_response(inventory=inventory, 
                              pre_filt=pre_filt, 
                              output=output_unit, 
                              water_level=60)
    
    return st_single[0]

# ==========================================
# Data Acquisition Functions
# ==========================================
def download_inventory_from_api(network, station_code):
    """Downloads XML Inventory from specified API."""
    url = f"http://202.90.198.40/sismon-seismic/inventory/{network}.{station_code}.xml"
    try:
        with urllib.request.urlopen(url) as response:
            inv_data = response.read()
            inv = read_inventory(io.BytesIO(inv_data))
            return inv
    except Exception as e:
        print(f"  ❌ Failed to get XML from API for station {station_code}: {e}")
        return None

# ==========================================
# Main Processing Pipeline
# ==========================================
def process_from_csv(csv_file, output_base_dir="processed_stations"):
    """Reads CSV, downloads data, and processes mbkm plots for each station."""
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: CSV file '{csv_file}' not found.")
        return

    df = pd.read_csv(csv_file)
    client = Client("https://geof.bmkg.go.id", user="mm", password="ina")
    
    for idx, row in df.iterrows():
        network = str(row['network']).strip()
        station = str(row['station']).strip()
        location = str(row.get('location', '*')).strip()
        channel = str(row.get('channel', 'SH?')).strip()
        start_dt = UTCDateTime(row['starttime'])
        end_dt = UTCDateTime(row['endtime'])

        # --- Parse Sensor Type ---
        raw_type = str(row.get('type', '')).strip().lower()
        if raw_type in ['', 'nan', 'none']:
            sensor_type = "seismometer"
        else:
            sensor_type = raw_type
            
        if sensor_type not in ["accelerometer", "seismometer"]:
            print(f"❌ Invalid type '{sensor_type}' for {station}. Must be 'accelerometer', 'seismometer', or blank. Skipping.")
            continue
            
        output_unit = "ACC" if sensor_type == "accelerometer" else "VEL"
        # -------------------------

        print(f"\n{'='*50}")
        print(f"📡 Processing Station: {station} ({idx + 1} of {len(df)})")
        print(f"⚙️  Sensor Type: {sensor_type.capitalize()} (Output: {output_unit})")
        print(f"{'='*50}")

        station_dir = os.path.join(output_base_dir, station)
        os.makedirs(station_dir, exist_ok=True)
        
        try:
            print("  -> Downloading Waveform (MSEED)...")
            st = client.get_waveforms(
                network=network,
                station=station,
                location=location,
                channel=channel,
                starttime=start_dt,
                endtime=end_dt
            )
            
            print("  -> Downloading Inventory (XML)...")
            inv = download_inventory_from_api(network, station)
            if inv is None:
                continue

            st.merge(method=1, fill_value='interpolate')
            st.trim(starttime=start_dt, endtime=end_dt)
            
            # 3. SAVE MSEED AND XML (Grouped by first 2 chars of channel)
            channel_prefixes = set([tr.stats.channel[:2] for tr in st])
            for prefix in channel_prefixes:
                st_prefix = st.select(channel=f"{prefix}*")
                inv_prefix = inv.select(channel=f"{prefix}*", 
                                        starttime=start_dt, 
                                        endtime=end_dt)
                
                loc_str = st_prefix[0].stats.location if st_prefix[0].stats.location else ""
                file_base = f"{network}.{station}.{loc_str}.{prefix}"
                mseed_path = os.path.join(station_dir, f"{file_base}.mseed")
                xml_path = os.path.join(station_dir, f"{file_base}.xml")
                
                st_prefix.write(mseed_path, format="MSEED")
                if inv_prefix:
                    inv_prefix.write(xml_path, format="STATIONXML")
                
                print(f"  -> Saved Grouped Data: {file_base}.mseed and .xml")
            
            # 4. Process and Plot Individual Traces
            for tr in st:
                trace_id = tr.id 
                print(f"\n  --- Processing Plot for Channel: {trace_id} ---")
                
                inv_filtered = inv.select(network=tr.stats.network, 
                                          station=tr.stats.station, 
                                          location=tr.stats.location, 
                                          channel=tr.stats.channel,
                                          starttime=start_dt,
                                          endtime=end_dt)
                
                if not inv_filtered:
                    print(f"  ⚠️ No matching active inventory for {trace_id} in this time range. Skipping plots.")
                    continue

                # Process the correction using dynamically determined output_unit (ACC or VEL)
                tr_corrected = correct_instrument_response(tr, inv_filtered, output_unit)
                
                # Execute plotting modules
                plot_instrument_response(inv_filtered, trace_id, station_dir, output_unit)
                plot_raw_waveform(tr, trace_id, station_dir)
                plot_corrected_waveform(tr_corrected, trace_id, station_dir, output_unit)
                
                # Pass sensor type to PPSD plotter
                plot_ppsd_obspy(tr, inv_filtered, trace_id, station_dir, sensor_type)
                
            print(f"✅ Successfully processed {station}")
            
        except Exception as e:
            print(f"❌ Failed processing station {station}: {e}")

if __name__ == "__main__":
    CSV_PATH = "input_stations.csv"
    process_from_csv(csv_file=CSV_PATH, output_base_dir="station_outputs")
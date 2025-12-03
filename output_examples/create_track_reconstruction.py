"""
create_track_reconstruction_table <out_prefix> [--tracksummary TRACKSUMMARY_ROOT] [--trackstates TRACKSTATES_ROOT]

Reads trackstates and tracksummary from CKF reconstruction output files,
and produces a per-measurement table showing the reconstructed track hits
similar to the hits table but for Kalman filter reconstructed tracks.

Output files (written to current working directory):
  <out_prefix>.txt   - textual summary and table of reconstructed measurements

Usage examples:
  python create_track_reconstruction_table ckf_table
  python create_track_reconstruction_table ckf_table --tracksummary tracksummary_ckf.root --trackstates trackstates_ckf.root

"""
import argparse
import sys
from pathlib import Path

# lazy imports (install if missing)
import uproot
import numpy as np

# Volume mapping (same as hits table for consistency)
VOLUME_MAP = {
    # barrel volumes
    8:  {'type': 'Pixel',      'region': 'barrel'},
    13: {'type': 'ShortStrip', 'region': 'barrel'},
    17: {'type': 'LongStrip',  'region': 'barrel'},
    # endcap volumes
    7:  {'type': 'Pixel',      'region': 'endcap'},
    12: {'type': 'ShortStrip', 'region': 'endcap'},
    9:  {'type': 'Pixel',      'region': 'endcap'},
    14: {'type': 'ShortStrip', 'region': 'endcap'},
    16: {'type': 'LongStrip',  'region': 'endcap'},
    18: {'type': 'LongStrip',  'region': 'endcap'},
}

def main():
    p = argparse.ArgumentParser(description='Create track reconstruction table from CKF output')
    p.add_argument('out_prefix', help='output prefix (files written as <prefix>.txt)')
    p.add_argument('--tracksummary', help='path to tracksummary_ckf.root (default: ./tracksummary_ckf.root)', default='tracksummary_ckf.root')
    p.add_argument('--trackstates', help='path to trackstates_ckf.root (default: ./trackstates_ckf.root)', default='trackstates_ckf.root')
    args = p.parse_args()

    out_prefix = Path(args.out_prefix)
    tracksummary_path = Path(args.tracksummary)
    trackstates_path = Path(args.trackstates)
    
    if not tracksummary_path.exists():
        print(f"Could not find tracksummary file: {tracksummary_path}")
        sys.exit(1)
        
    if not trackstates_path.exists():
        print(f"Could not find trackstates file: {trackstates_path}")
        sys.exit(1)

    # Open tracksummary file
    try:
        ts_file = uproot.open(str(tracksummary_path))
        ts_tree = ts_file['tracksummary']
    except Exception as e:
        print('Error opening tracksummary file:', e)
        sys.exit(1)

    # Open trackstates file  
    try:
        tst_file = uproot.open(str(trackstates_path))
        tst_tree = tst_file['trackstates']
    except Exception as e:
        print('Error opening trackstates file:', e)
        sys.exit(1)

    print(f"Processing {ts_tree.num_entries} tracks from CKF reconstruction...")

    # Read tracksummary data - note: data is nested by event, then by track
    event_numbers = ts_tree['event_nr'].array()
    track_numbers = ts_tree['track_nr'].array()
    n_measurements = ts_tree['nMeasurements'].array()
    n_states = ts_tree['nStates'].array()
    measurement_volumes = ts_tree['measurementVolume'].array()
    measurement_layers = ts_tree['measurementLayer'].array()
    measurement_chi2 = ts_tree['measurementChi2'].array()
    chi2_sum = ts_tree['chi2Sum'].array()
    ndf = ts_tree['NDF'].array()

    # Read trackstates data (positions of track states)
    ts_event_nr = tst_tree['event_nr'].array()
    ts_x = tst_tree['t_x'].array()
    ts_y = tst_tree['t_y'].array()
    ts_z = tst_tree['t_z'].array()
    ts_track_nr = tst_tree['track_nr'].array()
    
    # Check if we have layer information in trackstates
    has_layer_info = 'layer_id' in tst_tree.keys() and 'volume_id' in tst_tree.keys()
    if has_layer_info:
        ts_volume_id = tst_tree['volume_id'].array()
        ts_layer_id = tst_tree['layer_id'].array()
    else:
        ts_volume_id = None
        ts_layer_id = None

    # Create measurement rows by combining tracksummary and trackstates information
    all_rows = []
    track_summaries = []

    # Process each entry (each entry corresponds to an event in the original data)
    for entry_idx in range(len(track_numbers)):
        event_nr = np.atleast_1d(event_numbers[entry_idx])[0] if hasattr(event_numbers[entry_idx], '__len__') else event_numbers[entry_idx]
        event_tracks = track_numbers[entry_idx]
        event_n_meas = n_measurements[entry_idx] 
        event_n_states = n_states[entry_idx]
        event_meas_vols = measurement_volumes[entry_idx]
        event_meas_lays = measurement_layers[entry_idx]
        event_meas_chi2s = measurement_chi2[entry_idx]
        event_chi2_sum = chi2_sum[entry_idx]
        event_ndf = ndf[entry_idx]
        
        # Process each track in this entry
        for track_idx in range(len(event_tracks)):
            track_nr_within_event = int(event_tracks[track_idx])
            # Create unique track key from (event_nr, track_nr)
            track_key = (int(event_nr), track_nr_within_event)
            n_meas = int(event_n_meas[track_idx])
            meas_vols = event_meas_vols[track_idx]
            meas_lays = event_meas_lays[track_idx]
            meas_chi2s = event_meas_chi2s[track_idx]
            
            # Find corresponding trackstates for this track
            # Trackstates file has one entry per track, so we need to find the matching entry
            # by matching both event_nr and track_nr
            ts_idx = None
            for i in range(len(ts_event_nr)):
                if ts_event_nr[i] == event_nr and ts_track_nr[i] == track_nr_within_event:
                    ts_idx = i
                    break
            
            if ts_idx is None:
                print(f"Warning: No trackstates found for event {event_nr}, track {track_nr_within_event}")
                continue
            
            # Get track state data for this specific track
            event_ts_x = ts_x[ts_idx]
            event_ts_y = ts_y[ts_idx]
            event_ts_z = ts_z[ts_idx]
            
            # Convert to numpy arrays
            track_ts_x = np.atleast_1d(event_ts_x)
            track_ts_y = np.atleast_1d(event_ts_y)
            track_ts_z = np.atleast_1d(event_ts_z)
            
            if has_layer_info:
                event_ts_vol = ts_volume_id[ts_idx]
                event_ts_lay = ts_layer_id[ts_idx]
                track_ts_vol = np.atleast_1d(event_ts_vol)
                track_ts_lay = np.atleast_1d(event_ts_lay)

            # Track summary info
            track_summary = {
                'track_key': track_key,
                'n_measurements': n_meas,
                'n_states': int(event_n_states[track_idx]),
                'chi2_sum': float(event_chi2_sum[track_idx]),
                'ndf': int(event_ndf[track_idx]),
                'chi2_per_ndf': float(event_chi2_sum[track_idx]) / max(1, int(event_ndf[track_idx]))
            }
            track_summaries.append(track_summary)

            # Process measurements - match positions with volume/layer info
            # The trackstates contain positions for all states (predicted, filtered, smoothed)
            # We need to identify which correspond to measurements
            meas_count = 0
            for state_idx in range(len(track_ts_x)):
                x_pos = track_ts_x[state_idx]
                y_pos = track_ts_y[state_idx]
                z_pos = track_ts_z[state_idx]
                
                # Skip NaN positions (non-measurement states)
                if np.isnan(x_pos) or np.isnan(y_pos) or np.isnan(z_pos):
                    continue
                    
                # Try to match this measurement with the tracksummary info
                if meas_count < len(meas_vols):
                    volume_id = int(meas_vols[meas_count])
                    layer_id = int(meas_lays[meas_count])
                    chi2 = float(meas_chi2s[meas_count])
                else:
                    # Fallback to trackstates info if available
                    if has_layer_info and state_idx < len(track_ts_vol):
                        volume_id = int(track_ts_vol[state_idx])
                        layer_id = int(track_ts_lay[state_idx])
                    else:
                        volume_id = -1
                        layer_id = -1
                    chi2 = 0.0
                
                r_pos = np.sqrt(x_pos**2 + y_pos**2)
                
                row = {
                    'track_key': track_key,
                    'measurement_idx': meas_count,
                    'state_idx': state_idx,
                    'volume_id': volume_id,
                    'layer_id': layer_id,
                    'x_mm': float(x_pos),
                    'y_mm': float(y_pos),
                    'z_mm': float(z_pos),
                    'r_mm': float(r_pos),
                    'chi2': chi2
                }
                all_rows.append(row)
                meas_count += 1

    print(f"Processed {len(all_rows)} measurements from {len(track_summaries)} tracks")

    # Create mapping from (event_nr, track_nr) to sequential track numbers
    unique_track_keys = sorted(set(row['track_key'] for row in all_rows if 'track_key' in row))
    track_key_to_nr = {key: i+1 for i, key in enumerate(unique_track_keys)}
    
    # Assign sequential track numbers
    for row in all_rows:
        if 'track_key' in row:
            row['track_nr'] = track_key_to_nr[row['track_key']]
    
    # Update track summaries with sequential track numbers
    for summary in track_summaries:
        if 'track_key' in summary:
            summary['track_nr'] = track_key_to_nr[summary['track_key']]

    # Sort rows by track number, then by z position
    all_rows = sorted(all_rows, key=lambda r: (r['track_nr'], r['z_mm']))

    # Calculate statistics
    total_measurements = len(all_rows)
    unique_layers = set()
    vol_info = {}
    
    from collections import defaultdict
    per_vol_stats = defaultdict(lambda: {'measurements': 0, 'tracks': set(), 'layers': set()})
    
    for row in all_rows:
        vid = row['volume_id']
        if vid > 0:  # Skip invalid volume IDs
            unique_layers.add((vid, row['layer_id']))
            per_vol_stats[vid]['measurements'] += 1
            per_vol_stats[vid]['tracks'].add(row['track_nr'])
            per_vol_stats[vid]['layers'].add(row['layer_id'])
            
            # Get volume info
            if vid not in vol_info:
                if vid in VOLUME_MAP:
                    vm = VOLUME_MAP[vid]
                    vol_info[vid] = {
                        'detname': vm['type'] + (' Barrel' if vm['region'] == 'barrel' else ' Endcap'),
                        'type': vm['type'],
                        'region': vm['region']
                    }
                else:
                    vol_info[vid] = {
                        'detname': f'Unknown (vol {vid})',
                        'type': 'Unknown',
                        'region': 'unknown'
                    }

    # Generate output
    txt_path = Path(f"{out_prefix}.txt")
    with open(txt_path, 'w') as f:
        f.write("CKF Track Reconstruction Summary\n")
        f.write("=" * 40 + "\n\n")
        
        # Overall statistics
        f.write(f"Total reconstructed tracks: {len(track_summaries)}\n")
        f.write(f"Total measurements on tracks: {total_measurements}\n")
        f.write(f"Total unique layers hit: {len(unique_layers)}\n\n")
        
        # Per-track summaries
        f.write("Per-track summary:\n")
        for ts in track_summaries:
            f.write(f"  Track {ts['track_nr']}: {ts['n_measurements']} measurements, "
                   f"{ts['n_states']} states, χ²/NDF = {ts['chi2_per_ndf']:.3f}\n")
        f.write("\n")
        
        # Per-volume statistics  
        f.write("Per-volume statistics:\n")
        for vid in sorted(per_vol_stats.keys()):
            if vid > 0:
                stats = per_vol_stats[vid]
                info = vol_info[vid]
                f.write(f"  Volume {vid} ({info['detname']}): {stats['measurements']} measurements, "
                       f"{len(stats['layers'])} layers, {len(stats['tracks'])} tracks\n")
        f.write("\n")

        # Detailed measurement table
        f.write("Detailed measurement table (sorted by track, then z):\n")
        hdr = ('trk', 'vol', 'lay', 'r_mm', 'x_mm', 'y_mm', 'z_mm', 'χ²')
        widths = [4, 4, 4, 9, 9, 9, 9, 8]
        
        # Header
        header_line = ' '.join(h.center(w) for h, w in zip(hdr, widths))
        f.write(header_line + '\n')
        f.write('-' * sum(widths) + '\n')
        
        # Data rows
        for row in all_rows:
            if row['volume_id'] > 0:  # Only show valid measurements
                parts = [
                    str(row['track_nr']).rjust(widths[0]),
                    str(row['volume_id']).rjust(widths[1]),
                    str(row['layer_id']).rjust(widths[2]),
                    f"{row['r_mm']:.1f}".rjust(widths[3]),
                    f"{row['x_mm']:.1f}".rjust(widths[4]),
                    f"{row['y_mm']:.1f}".rjust(widths[5]),
                    f"{row['z_mm']:.1f}".rjust(widths[6]),
                    f"{row['chi2']:.3f}".rjust(widths[7]),
                ]
                f.write(' '.join(parts) + '\n')

    print(f"CKF reconstruction table written to: {txt_path}")

if __name__ == '__main__':
    main()
import os

def check_readiness():
    # Define the expected structure based on the README.md inventory
    inventory = {
        "gmst": [
            "berkEarth_globalmean_airTaboveseaice.txt",
            "nasa_GLB.Ts+dSST.csv",
            "hadcrut5_global_monthly.csv",
            "noaa_atmoT_landocean.txt"
        ],
        "gmslr": [
            "nasa_GMSL_TPJAOS_5.2.txt",
            "frederikse2020_global_timeseries.xlsx",
            "dangendorf2024_KalmanSmootherHR_Global.nc",
            "horwath2021_ESACCI_SLBC_v2.2.csv"
        ],
        "glaciers": [
            "0_global_glambie_consensus.csv"
        ],
        "ice_sheets/greenland": [
            "imbie_greenland_2021_mm.csv"
        ],
        "ice_sheets/antarctica": [
            "imbie_antarctica_2021_mm.csv",
            "imbie_west_antarctica_2021_mm.csv",
            "imbie_east_antarctica_2021_mm.csv",
            "imbie_antarctic_peninsula_2021_mm.csv"
        ],
        "tws": [
            "GRCTellus.JPL.200204_202510.GLO.RL06.3M.MSCNv04CRI.nc"
        ],
        "saod": [
            "GloSSAC_V2.2.nc",
            "maunaLoa_transmission.txt"
        ],
        "enso": [
            "noaa_mei_index.txt",
            "noaa_oceanicNinoIndex_oni.csv"
        ]
    }

    base_path = os.path.join("data", "raw")
    missing_count = 0
    found_count = 0

    print(f"{'='*60}")
    print(f"{'DATA READINESS REPORT':^60}")
    print(f"{'='*60}")

    for folder, files in inventory.items():
        print(f"\n📂 Checking [data/raw/{folder}]...")
        folder_path = os.path.join(base_path, folder)
        
        if not os.path.exists(folder_path):
            print(f"  ❌ FOLDER MISSING: {folder_path}")
            missing_count += len(files)
            continue

        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.exists(file_path):
                print(f"  ✅ {file}")
                found_count += 1
            else:
                print(f"  ❌ MISSING: {file}")
                missing_count += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Total Expected Files: {found_count + missing_count}")
    print(f"  Files Found:         {found_count}")
    print(f"  Files Missing:       {missing_count}")
    print(f"{'='*60}")

    if missing_count == 0:
        print("\n🚀 ALL SYSTEMS GO: You are ready for the DOLS calibration.")
    else:
        print("\n⚠️  ACTION REQUIRED: Please move the missing files to the paths listed above.")

if __name__ == "__main__":
    check_readiness()

"""
Script to update main.ipynb to work with the new all_data_combined structure.
Run this script once to apply the changes, then you can delete it.
"""

import json
import re

def main():
    notebook_path = "main.ipynb"
    
    # Read the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    changes_made = 0
    
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        
        source = cell["source"]
        source_str = "".join(source)
        
        # Change 1: Update DATA_DIR to DATA_FILE
        if 'DATA_DIR = os.path.join(' in source_str and '"all_data"' in source_str:
            new_source = [
                "import os\n",
                "\n",
                "# Path to the combined CSV file\n",
                "DATA_FILE = os.path.join(\n",
                '    "all_data_combined",\n',
                '    f"data_{settings[\'market\']}_{settings[\'interval\']}.csv"\n',
                ")\n",
                "\n",
                "SETTINGS_FILE = (\n",
                '    "candle_settings_1week.json"\n',
                '    if settings["interval"] == "1week"\n',
                '    else "candle_settings_1d.json"\n',
                ")\n",
            ]
            cell["source"] = new_source
            changes_made += 1
            print("[OK] Updated DATA_DIR -> DATA_FILE")
        
        # Change 2: Update load_symbol_data function
        if "def load_symbol_data(symbol, data_dir, window):" in source_str:
            new_source = [
                'def load_symbol_data(symbol, data_file, window):\n',
                '    """\n',
                '    Load data for a single symbol from a combined CSV file.\n',
                '    \n',
                '    Args:\n',
                '        symbol: The ticker symbol to load\n',
                '        data_file: Path to the combined CSV file\n',
                '        window: Number of rows to return at the end\n',
                '    """\n',
                '    global SYMBOLS\n',
                '\n',
                '    if not os.path.isfile(data_file):\n',
                '        print(f"❌ Combined data file not found: {data_file}")\n',
                '        return None\n',
                '\n',
                '    # Read combined CSV and filter by symbol\n',
                '    df_all = pd.read_csv(data_file, skiprows=[1])\n',
                '    \n',
                '    # Filter for the specific symbol\n',
                '    df = df_all[df_all["Symbol"] == symbol].copy()\n',
                '    \n',
                '    if df.empty:\n',
                '        if symbol in SYMBOLS:\n',
                '            SYMBOLS.remove(symbol)\n',
                '        print(f"❌ Removed (not in data file): {symbol}")\n',
                '        return None\n',
                '\n',
                '    # ---- normalize datetime column name ----\n',
                '    if "Datetime" in df.columns and "Date" not in df.columns:\n',
                '        df.rename(columns={"Datetime": "Date"}, inplace=True)\n',
                '\n',
                '    REQUIRED = ["Date", "Open", "High", "Low", "Close", "Volume"]\n',
                '    if not all(col in df.columns for col in REQUIRED):\n',
                '        if symbol in SYMBOLS:\n',
                '            SYMBOLS.remove(symbol)\n',
                '        print(f"❌ Removed (bad schema): {symbol}")\n',
                '        return None\n',
                '\n',
                '    df = df[REQUIRED]\n',
                '\n',
                '    # ---- datetime handling ----\n',
                '    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")\n',
                '    df.set_index("Date", inplace=True)\n',
                '    df.sort_index(inplace=True)\n',
                '\n',
                '    df["Weekday"] = df.index.day_name()  \n',
                '\n',
                '    # ---- numeric conversion ----\n',
                '    for col in ["Open", "High", "Low", "Close"]:\n',
                '        df[col] = pd.to_numeric(df[col], errors="coerce")\n',
                '\n',
                '    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)\n',
                '    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)\n',
                '    \n',
                '    return df.tail(window)\n',
            ]
            cell["source"] = new_source
            changes_made += 1
            print("[OK] Updated load_symbol_data function")
        
        # Change 3: Update the call to load_symbol_data
        if "load_symbol_data(symbol, DATA_DIR, window=window)" in source_str:
            new_source_str = source_str.replace(
                "load_symbol_data(symbol, DATA_DIR, window=window)",
                "load_symbol_data(symbol, DATA_FILE, window=window)"
            )
            # Split back into lines for the notebook format
            cell["source"] = [line + "\n" if not line.endswith("\n") and i < len(new_source_str.split("\n")) - 1 
                             else line 
                             for i, line in enumerate(new_source_str.split("\n"))]
            # Fix: properly format as list of lines
            lines = new_source_str.split("\n")
            cell["source"] = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
            changes_made += 1
            print("[OK] Updated load_symbol_data call: DATA_DIR -> DATA_FILE")
    
    # Write the updated notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"\nDone! Made {changes_made} changes to {notebook_path}")
    print("You can now delete this script (update_notebook.py)")

if __name__ == "__main__":
    main()

# SSB Modulation and Demodulation in FDM System

This project implements Single Sideband (SSB) modulation and demodulation for three speech signals in a Frequency-Division Multiplexing (FDM) system using Python.

## Project Structure
- `data/input/`: Contains raw input audio files.
- `data/filtered/`: Contains filtered audio files after applying a low-pass filter.
- `data/output/`: Contains final demodulated audio files.
- `src/`: Contains Python scripts for audio recording, filtering, modulation, and demodulation.
- `docs/`: Contains project documentation and report.
- `README.md`: Provides project overview and instructions.
- `requirements.txt`: Lists project dependencies.

## How to Run
1. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Record Audio**:
    ```bash
    python src/record_audio.py
    ```

3. **Filter Audio**:
    ```bash
    python src/filter_audio.py
    ```

4. **Run the Modulation Script**:
    ```bash
    python src/modulate.py
    ```

5. **Run the Demodulation Script**:
    ```bash
    python src/demodulate.py
    ```

## Requirements
- Python 3.8 or higher
- NumPy
- SciPy
- Matplotlib

## Documentation
The project documentation and report can be found in the `docs/` directory.

Read the full documentation [here](docs/Project%20Documentation.pdf).

## License
This project is licensed under the MIT License.


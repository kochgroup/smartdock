# SMARTDock: A Toolkit for Automated Target-Specific Scoring Functions

**SMARTDock** (Scoring with Machine learning and Activity for Ranking Targeted Docking) is an integrated, automated workflow designed to enhance virtual screening in structure-based drug discovery. It develops **target-specific scoring functions (STSF)** by integrating publicly available bioactivity data (ChEMBL), molecular docking (using **GOLD**), and machine learning (ML) classification models.

The workflow uses the **PADIF** (Protein per Atom Score Contributions Derived Interaction Fingerprint) representation to encode key protein-ligand interactions and classify likely binders.

-----

## 1\. Prerequisites

To run SMARTDock, you must have the following installed locally, as the Docker container relies on mounting and accessing these files:

  * **Docker:** Required for running the containerized workflow.
  * **GOLD Docking Software:** A local installation of the GOLD software (part of the CCDC suite).
  * **CCDC License:** An active license key is required for the CCDC Python API and GOLD to function within the container.

-----

## 2\. Installation and Setup

### A. Clone the Repository

Clone the project repository and navigate into the directory:

```bash
git clone https://github.com/kochgroup/smartdock.git
cd smartdock
```

### B. Configure CCDC Paths (`compose.yml`)

You must edit the **`compose.yml`** file to match your local CCDC installation paths.

1.  **Mount Path:** Change the local volume path `/appl/ccdc/` to the actual path of your CCDC installation directory.

      * **Original:** `- /appl/ccdc/:/mnt/ccdc`
      * **Action:** Replace `/appl/ccdc/` with your CCDC root directory path.

2.  **Environment Variables:** Verify that the `CSDHOME`, `GOLD_DIR`, and other CCDC environment variables match the **version and paths** of your mounted CCDC installation.

      * *Example based on your file:* The current paths are set for a `CSDS2020` installation. Adjust `2020` if you use a different version.
        ```yaml
        environment:
          - CSDHOME=/mnt/ccdc/CSDS2020/CSD_2020
          - GOLD_DIR=/mnt/ccdc/CSDS2020/Discovery_2020/GOLD
          # ... other CCDC variables
        ```

### C. Activate License (`padif_start.sh`)

Edit the **`padif_start.sh`** file to use your actual CCDC license key.

  * **Action:** Replace the placeholder key `96DAD1-E0657B-44F7B2-E17D11-EB9203-EEBCCE` with your valid CCDC license key.

    ```bash
    # Replace the key below with your valid CCDC license key
    docker exec -it padif_app /bin/bash -c "/mnt/ccdc/CSDS2020/CSD_2020/bin/ccdc_activator -a -k YOUR_CCDC_LICENSE_KEY_HERE"
    ```

### D. Start the Environment

Make the start script executable and run it to build the image (if necessary) and launch the container:

```bash
chmod +x padif_start.sh
./padif_start.sh
```

The script will:

1.  Run `docker compose up -d` to build the `padif_app` image and start the container in the background.
2.  Execute the CCDC activator command with your license key inside the container.
3.  Attach you to a bash session inside the running container, placed in the `/code/work` directory, ready to run the SMARTDock script.

-----

## 3\. Running SMARTDock (Training and Screening)

Once inside the container's shell, you can execute the main SMARTDock script (assuming the entry script is named `smartdock.py` or similar).

### A. Create an Input Directory

Ensure your input files are placed in the local `./data` folder before starting the container, as this folder is mounted to `/code/work/data` inside the container.

### B. Core Command Syntax

The workflow is typically executed with a single command providing the necessary inputs:

```bash
# Example command (adjust script name and arguments as needed)
python3 /code/src/smartdock_workflow.py \
    --chembl-id CHEMBL5567 \
    --pdb-file ./data/3IES.pdb \
    --smiles-list ./data/test_compounds.smi \
    --model XGBoost \
    --splitting scaffold
```

### C. Required Inputs

1.  **`--chembl-id`:** The ChEMBL ID of the target protein.
2.  **`--pdb-file`:** Path to the PDB file of the protein structure (should be placed in the mounted `./data` folder).
3.  **`--smiles-list`:** Path to a file containing test compounds (SMILES, ID format) for external screening.

### D. Optional Parameters

| Parameter | Options | Description |
| :--- | :--- | :--- |
| **`--model`** | RF, XGBoost, MLP | Select the machine learning algorithm(s) for the scoring function. |
| **`--splitting`** | random, scaffold | Select the data splitting method for model validation and training. |
| **`--search-efficiency`** | Integer (e.g., 200) | GOLD search efficiency parameter. |
| **`--ligand-id`** | String (e.g., 'LIG') | Co-crystallized ligand ID to define the binding site. |

-----

## 4\. Software and Dependencies

The Docker image is built on Python 3.9. Key dependencies installed via the **`requirements.txt`** file include:

  * `deepchem`
  * `chembl_webresource_client`
  * `rdkit`
  * `pycaret`
  * `xgboost`
  * `csd-python-api` (installed via a custom CCDC index)
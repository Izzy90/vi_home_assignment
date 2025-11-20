## User Guide

Follow the steps below to recreate the project environment with `uv`.

1. **Create the virtual environment**
   - Run `uv venv` from the repository root. This reads `pyproject.toml` and provisions a `.venv` folder with Python 3.11+.
2. **Activate the environment**
   - macOS/Linux: `source .venv/bin/activate`
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
3. **Install the project requirements**
   - With the virtual environment active, run `uv pip install -r pyproject.toml` to install the dependencies declared in `pyproject.toml`. If you prefer deterministic installs from `uv.lock`, run `uv pip sync`.

To run the main script, use: `python -m main`

### Runtime options

- `--min-recall`: minimum recall threshold enforced when searching for the precision-maximizing cutoff (default `0.05`, configurable in `config.py`). Example:
  ```
  python -m main --min-recall 0.1
  ```

## Solution Description

After choosing a minimum recall (what percent of churn we want to fix), the model will create a list of N members that are most likely to churn. The list can be found in `outputs\top_N_member_ids_test.csv`

### Feature Selection
In order to create my features, I came up with features that made logical sense to me that should have decent predictive power on churn, like app usage statistics, personal interests (through web-visits) and personal medical conditions (with icd-codes). I didn't go very deep in this part and it's definitely one that I would pursue in a future iteration of this project, as it could have a great impact on performance.

### Model Evaluation
I started by deciding on the business KPI for this project, which is outreach ROI. The formula to compute this ROI is as follows:
ROI = (N_s * OG) / (N_t * OC)
Where:
N_t - total members in outreach list
OC - Outreach Cost
OG - Outreach Gain
N_s - number of members “saved” by outreach (churn prevented)
Assuming a member that was going to churn will be “saved” by outreach, we get: 
N_s = N_t * Pr –> ROI = Pr * (OG/OC)
Where Pr is the precision of a classifier that predicts the probability of churn. Since OG and OC are both constant, in order to maximise ROI we should find a value for N that maximizes precision. So, precision will be our main evaluation metric.
While sampling the hyperparameter space for my XGBoost models, I kept track of the hyperparameter combination with the best precision.
Also, that's how I chose the threshold for determining which n members get outreach. More on that in the last chapter. 

### Using Outreach Data in Modelling
There's an underlying research question in this assignment that I didn't have the time to address, and that is how does outreach affect churn. Since the training data has flags for both, this could and should be done, because if we are able to successfully model it, we could avoid reaching out to members that are going to churn regardless of whether or not we reach out to them. If I had more time to work on this problem, I would definitely tackle this question, intuitively using causal inference tools like DoWhy for example.
Therefore, I chose to narrow the scope to only predicting the probability of churn for each member. And because I didn't want to pollute my training data with datapoints that were affected by outreach, I chose to filter out all training datapoints with outreach=1. That was ~40% of training data which is a serious compromise but it had to be done.

### Selecting n (Outreach Size)
In order to select the outreach size n, my method was:
1. Train a bunch of XGBoost classifiers on the extracted features (sweeping over the hyperparameter space to find the best combination).
2. Given the min_recall parameter, find the probability threshold that maximizes precision.
3. Use the best performing model to predict probability of churn over test data.
4. Use the threshold from step 2 to recommend outreach for all test-set members with a higher probability of churn than the threshold. The number of such members is n. The expectation is that this would maximize precision, and thus also outreach ROI.
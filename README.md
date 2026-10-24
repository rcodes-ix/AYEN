# AYEN: Supply Chain Disruption Intelligence

**AI-powered shipment risk assessment and supply chain decision support, with a focus on Ethiopian logistics corridors.**

**Live application:** https://ayen-supply-chain.streamlit.app/

AYEN is an AI-powered supply chain intelligence application designed to help manufacturers assess potential shipment delays, explore historical shipment data, and make more informed logistics and inventory decisions.

The application combines machine learning-based delay prediction, interactive shipment analytics, operational recommendations, and individual prediction explanations. It supports a general model trained on a synthetic shipment dataset and manufacturer-specific models trained using uploaded historical shipment records.

AYEN also incorporates simulated Ethiopian corridor-related indicators, including Djibouti port dwell time, Modjo dry port dwell time, and foreign exchange (FX) availability.

AYEN is currently a **minimum viable product (MVP)** intended for demonstration, experimentation, and further development. Its included datasets are synthetic. The corridor indicators are simulated rather than live operational measurements, and the model's predictions have not been validated against real-world shipment outcomes.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How AYEN Works](#how-ayen-works)
- [Application Pages](#application-pages)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [General Model Training](#general-model-training)
- [Manufacturer-Specific Models](#manufacturer-specific-models)
- [Using the Demonstration Dataset](#using-the-demonstration-dataset)
- [Dataset Schema](#dataset-schema)
- [Model Evaluation](#model-evaluation)
- [Risk Interpretation](#risk-interpretation)
- [Data Storage](#data-storage)
- [Configuration and Customization](#configuration-and-customization)
- [Troubleshooting](#troubleshooting)
- [Limitations and Security Considerations](#limitations-and-security-considerations)
- [Future Development](#future-development)
- [License](#license)

## Overview

Supply chain disruptions can affect production schedules, inventory availability, transportation costs, and customer commitments. Manufacturers need ways to identify potential risks before they become operational problems.

AYEN provides a starting point for analyzing shipment-related risk using information such as:

- Supplier reliability
- Planned shipment lead time
- Transport mode
- Port congestion
- Weather severity
- Customs delay risk
- Inventory coverage
- Historical supplier delay rate
- Shipment cost
- Product criticality
- Djibouti port dwell time
- Modjo dry port dwell time
- FX availability

The general model uses shipment-related features to estimate the probability of a delay. The application then presents a risk category, relevant operational recommendations, and the top three input factors that most influenced the prediction according to its sensitivity-based explanation method.

AYEN supports two modeling workflows:

1. **General AYEN model:** A baseline model trained on the general synthetic shipment dataset.
2. **Manufacturer-specific model:** A separate model trained using a manufacturer's uploaded historical shipment records.

The manufacturer-specific workflow allows predictions to use patterns learned from a manufacturer's own historical data rather than relying exclusively on the general demonstration dataset.

## Key Features

### 1. Shipment disruption risk prediction

Users can enter shipment information and receive a model-generated estimate of delay probability.

The application displays the estimated probability, assigns a risk category, and provides recommendations based on the prediction and available shipment characteristics.

### 2. Ethiopian corridor-related features

The general dataset and prediction workflow incorporate three additional simulated indicators:

- **Djibouti port dwell time:** Represents the time a shipment spends at the Djibouti port.
- **Modjo dry port dwell time:** Represents time spent at the Modjo dry port.
- **FX availability index:** Represents an assumed level of foreign exchange availability that may affect logistics or import-related operations.

These features make the demonstration more relevant to Ethiopian import and logistics scenarios. They are synthetic indicators, not measurements obtained from live port, customs, banking, or logistics systems.

### 3. Top-three prediction explanations

For a general-model assessment, AYEN provides a ranked explanation of three input features associated with changes in the predicted delay probability.

The explanation method changes one feature at a time to a reference value and observes how the prediction changes. Features associated with larger changes are ranked more highly.

This helps users investigate why a particular shipment received its predicted risk level.

These explanations are sensitivity-based, not proof of causation. They should not be interpreted as definitive statements about why a real shipment will be delayed.

### 4. General and manufacturer-specific models

Users can select between the general AYEN model and an available trained manufacturer-specific model.

Manufacturer-specific models are trained independently using each manufacturer's saved historical shipment dataset.

### 5. Manufacturer profiles and historical data

Users can create manufacturer profiles and upload historical shipment records.

The application validates uploaded data before saving it for analysis and model training.

### 6. Shipment analytics and exploration

The application provides an overview of shipment data through summary metrics and interactive charts. Users can explore shipment records and examine patterns in the selected dataset.

### 7. Model transparency

The application displays available evaluation metrics for the general model or a selected manufacturer-specific model.

Users can inspect model performance information while considering the limitations of the evaluation data.

### 8. Operational recommendations

AYEN can suggest actions involving:

- Supplier communication
- Inventory and safety-stock review
- Shipment tracking
- Port and dry port monitoring
- Customs documentation
- Foreign exchange availability checks
- Production continuity planning

Recommendations depend on the available shipment information and predicted risk. They are decision-support suggestions, not guarantees that a particular action will prevent a disruption.

---

## Ethiopian Corridor Intelligence

Ethiopia relies on cross-border logistics for many imported goods and materials. Delays across ports, inland transport routes, dry ports, customs processes, and import-related financial operations can affect shipment lead times and inventory planning.

AYEN's current demonstration incorporates three corridor-related features to explore how these conditions might be represented in a shipment risk model.

| Feature                    | Purpose                                        | Current data source      |
| -------------------------- | ---------------------------------------------- | ------------------------ |
| `djibouti_port_dwell_days` | Represents time spent at Djibouti port         | Synthetic data generator |
| `modjo_dwell_days`         | Represents time spent at Modjo dry port        | Synthetic data generator |
| `fx_availability_index`    | Represents an assumed level of FX availability | Synthetic data generator |

The general model uses these indicators alongside supplier, shipment, inventory, and other logistics features.

**Important:** These indicators are generated for demonstration purposes. They are not live measurements of current port congestion, actual cargo dwell times, or Ethiopia's foreign exchange availability. Their inclusion does not establish that the model accurately predicts real Ethiopian supply chain disruptions.

Future development could connect these features to reliable operational data sources and evaluate their predictive value using verified shipment outcomes.

---

## How AYEN Works

### General prediction workflow

1. Load the general synthetic shipment dataset.
2. Load the trained general model and its evaluation metrics, when available.
3. Allow users to explore shipment records or enter details about a new shipment.
4. Process the entered features using the preprocessing pipeline associated with the selected model.
5. Generate a predicted delay probability.
6. Assign a low, medium, or high risk category.
7. Generate a sensitivity-based explanation of the top three input factors, where supported.
8. Display recommendations based on the predicted risk and shipment characteristics.

### Manufacturer-specific training workflow

1. Create a manufacturer profile.
2. Upload historical shipment records containing known delay outcomes.
3. Validate the uploaded dataset.
4. Save the validated history.
5. Split the records into training and evaluation subsets.
6. Train a model and evaluate it on the held-out subset.
7. Train the final model using all available records.
8. Save the trained model and its evaluation metrics.
9. Select the manufacturer-specific model when assessing a new shipment.

The manufacturer-specific workflow uses its own feature schema. It is separate from the general model's corridor-related feature additions.

---

## Application Pages

AYEN contains five primary pages.

### Overview

Provides summary statistics and charts for the selected dataset.

Users can choose the general AYEN dataset or a manufacturer's saved shipment history. The displayed results depend on the selected data source.

### New Assessment

Provides a form for assessing the disruption risk of a shipment.

Users can select the general model or an available manufacturer-specific model and enter the required shipment information.

For the general model, the form includes the additional corridor-related indicators.

The application returns a predicted delay probability, a risk category, recommendations, and a ranked explanation of the top three input factors where supported by the selected prediction workflow.

A prediction is an estimate based on learned patterns. It is not confirmation that a shipment will or will not be delayed.

### Shipment Explorer

Provides an interface for exploring shipment records.

Users can select the general dataset or a manufacturer's saved history to investigate shipment characteristics and patterns.

### Data Import

Supports manufacturer profile creation, historical dataset upload, data validation, history storage, and manufacturer-specific model training.

A dataset must contain the required feature columns and known delay outcomes.

### Model Transparency

Displays available evaluation metrics for the general model or a selected trained manufacturer-specific model.

This page helps users inspect model evaluation results and understand the limitations of the available evidence.

---

## Technology Stack

| Technology     | Purpose                                                                 |
| -------------- | ----------------------------------------------------------------------- |
| Python         | Application logic, data generation, and model training                  |
| Streamlit      | Interactive web application                                             |
| Pandas         | Dataset loading, validation, and analysis                               |
| NumPy          | Numerical operations and synthetic data generation                      |
| Scikit-learn   | Preprocessing pipelines, classification, model training, and evaluation |
| Joblib         | Saving and loading trained models                                       |
| Plotly Express | Interactive charts                                                      |

---

## Project Structure

The main project files are organized as follows:

```text
AYEN/
├── app.py
├── manufacturer_models.py
├── train_model.py
├── generate_data.py
├── generate_manufacturer_data.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── ayen_synthetic_shipments.csv
│   ├── manufacturer_demo.csv
│   └── manufacturers/
│       └── <manufacturer_id>/
│           ├── history.csv
│           ├── model.joblib
│           └── metrics.json
│
└── models/
    ├── disruption_model.joblib
    └── model_metrics.json
```

This tree illustrates the expected layout. Generated datasets, model artifacts, and manufacturer directories may not exist in a fresh checkout. The application and scripts create generated directories when needed.

### File descriptions

**`app.py`**

Contains the Streamlit application, navigation, shipment assessment interface, dataset selection, visualizations, prediction explanations, recommendations, and model evaluation views.

**`manufacturer_models.py`**

Contains the functions used to validate manufacturer datasets, save and load shipment histories, train manufacturer-specific models, and load their evaluation metrics.

**`train_model.py`**

Trains and evaluates the general AYEN model. Saves the trained model and its evaluation metrics under `models/`.

**`generate_data.py`**

Generates the general synthetic shipment dataset, including the simulated Ethiopian corridor-related features.

**`generate_manufacturer_data.py`**

Generates a synthetic manufacturer dataset intended to demonstrate the upload and model-training workflow.

**`requirements.txt`**

Lists the Python dependencies required by the application.

**`data/ayen_synthetic_shipments.csv`**

Contains the general synthetic shipment dataset used by the baseline workflow.

**`data/manufacturer_demo.csv`**

Contains synthetic records suitable for demonstrating manufacturer-specific training.

**`models/disruption_model.joblib`**

Contains the serialized general model created by `train_model.py`.

**`models/model_metrics.json`**

Contains evaluation metrics for the general model.

**`data/manufacturers/`**

Stores manufacturer-specific histories, trained models, and evaluation metrics on the local filesystem.

---

## Requirements

Before running AYEN, ensure that you have:

- Python 3.10 or another Python version compatible with the dependencies.
- `pip` installed.
- Access to a terminal.
- A Python virtual environment.
- The project files.

AYEN can run locally on Linux, WSL Ubuntu, or another compatible Python environment.

## Installation

### 1. Open the project directory

In a WSL terminal, run:

```bash
cd ~/AYEN
```

Confirm that the project files are present:

```bash
ls
```

You should see files such as `app.py`, `train_model.py`, `manufacturer_models.py`, and `requirements.txt`.

### 2. Activate a virtual environment

If you already have a working virtual environment, activate it using its actual path.

For example:

```bash
source ~/myenv/bin/activate
```

Alternatively, create a new environment inside the project:

```bash
python -m venv .venv
source .venv/bin/activate
```

If your virtual environment is already active, you do not need to activate it again.

### 3. Install dependencies

Run:

```bash
python -m pip install -r requirements.txt
```

This installs the required packages into the active Python environment.

If installation fails, check the error message and confirm that your Python version is compatible with the dependencies.

---

## Running the Application

From the project directory, generate the general dataset and train the general model:

```bash
python generate_data.py
python train_model.py
```

Then start Streamlit:

```bash
streamlit run app.py
```

Streamlit will display a local address in the terminal. Open that address in your browser.

To stop the application, return to the terminal and press `Ctrl+C`.

If you are working with an existing trained model and dataset, regeneration and retraining are not necessarily required every time you launch the application.

---

## General Model Training

The general model uses the dataset at:

```text
data/ayen_synthetic_shipments.csv
```

To regenerate the dataset and train the model, run:

```bash
python generate_data.py
python train_model.py
```

The general training script:

1. Loads the general shipment dataset.
2. Checks that the required features and target column exist.
3. Separates input features from the delay outcome.
4. Splits the data into training and testing subsets.
5. Builds preprocessing pipelines for numeric and categorical variables.
6. Trains a Random Forest classifier.
7. Calculates evaluation metrics on the held-out test subset.
8. Saves the trained model and metrics.

The general model includes the corridor-related indicators:

- `djibouti_port_dwell_days`
- `modjo_dwell_days`
- `fx_availability_index`

The saved artifacts are:

```text
models/disruption_model.joblib
models/model_metrics.json
```

The trained pipeline includes preprocessing steps so that numeric and categorical features are handled consistently during training and prediction.

Regenerating the dataset replaces the existing generated dataset. Retraining replaces the existing general model and its metrics.

**Note:** The general model's feature schema differs from the manufacturer-specific model's schema. Keep these workflows separate when modifying the code.

---

## Manufacturer-Specific Models

AYEN supports training a separate model for each manufacturer using its saved shipment history.

### Step 1: Open Data Import

Start the application:

```bash
streamlit run app.py
```

Navigate to the **Data Import** page.

### Step 2: Create a manufacturer profile

Enter a manufacturer name and create its profile through the application interface.

Use a consistent name for each manufacturer so that its records remain associated with the intended profile.

### Step 3: Upload historical shipment records

Upload a CSV containing the required manufacturer-specific features and known shipment outcomes.

The application validates the data before saving it.

### Step 4: Save the history

Save the validated dataset to the manufacturer profile.

The history is stored locally under a directory similar to:

```text
data/manufacturers/<manufacturer_id>/history.csv
```

### Step 5: Train the manufacturer-specific model

Select the manufacturer and use the model-training control on the Data Import page.

The application evaluates the model on a held-out subset and then trains the final model using the complete saved dataset.

The generated files are stored under the manufacturer's directory:

```text
data/manufacturers/<manufacturer_id>/model.joblib
data/manufacturers/<manufacturer_id>/metrics.json
```

### Step 6: Use the trained model

After training, the manufacturer can be selected in the relevant application pages.

- **New Assessment:** Select the manufacturer-specific model to assess a shipment.
- **Overview:** Select the manufacturer's history to view its available analytics.
- **Shipment Explorer:** Explore the manufacturer's saved shipment records.
- **Model Transparency:** Inspect the manufacturer's model evaluation metrics.

Manufacturer-specific models are trained independently. Training one manufacturer's model does not automatically retrain the general model or other manufacturers' models.

---

## Using the Demonstration Dataset

The project includes:

```text
data/manufacturer_demo.csv
```

This file contains synthetic shipment records and can be used to test the manufacturer-specific workflow without preparing a dataset first.

To use it:

1. Launch AYEN.
2. Open Data Import.
3. Create a test manufacturer profile.
4. Upload `data/manufacturer_demo.csv`.
5. Save the history.
6. Train the manufacturer's model.
7. Open New Assessment and select the trained manufacturer.
8. Inspect the results and open Model Transparency to review the evaluation metrics.

To regenerate the manufacturer demonstration dataset, run:

```bash
python generate_manufacturer_data.py
```

This replaces the existing demonstration CSV with a newly generated synthetic dataset.

The demonstration data is fabricated for development and demonstration. It is not real manufacturer shipment history and must not be presented as evidence of real-world predictive performance.

---

## Dataset Schemas

AYEN uses two different feature schemas: one for the general model and another for manufacturer-specific models.

### General model features

The general model uses the following input features.

| Feature                    | Type        | Description                             |
| -------------------------- | ----------- | --------------------------------------- |
| `order_quantity`           | Numeric     | Quantity ordered                        |
| `planned_lead_time_days`   | Numeric     | Planned shipment lead time              |
| `supplier_reliability`     | Numeric     | Supplier reliability indicator          |
| `port_congestion`          | Numeric     | Port congestion indicator               |
| `weather_severity`         | Numeric     | Weather severity indicator              |
| `customs_delay_risk`       | Numeric     | Customs delay risk indicator            |
| `inventory_days`           | Numeric     | Inventory coverage in days              |
| `historical_delay_rate`    | Numeric     | Historical delay rate                   |
| `shipment_cost`            | Numeric     | Shipment cost                           |
| `djibouti_port_dwell_days` | Numeric     | Simulated dwell time at Djibouti port   |
| `modjo_dwell_days`         | Numeric     | Simulated dwell time at Modjo dry port  |
| `fx_availability_index`    | Numeric     | Simulated FX availability indicator     |
| `supplier_country`         | Categorical | Supplier country                        |
| `transport_mode`           | Categorical | Shipment transport mode                 |
| `product_category`         | Categorical | Product or material category            |
| `criticality`              | Categorical | Operational criticality of the shipment |

The general model's dataset may also contain additional descriptive or outcome columns. Only the features configured for model training are used as model inputs.

### Manufacturer-specific model features

Manufacturer uploads must contain the following 14 columns.

| Column                   | Type           | Description                                     |
| ------------------------ | -------------- | ----------------------------------------------- |
| `order_quantity`         | Numeric        | Quantity ordered                                |
| `planned_lead_time_days` | Numeric        | Planned shipment lead time in days              |
| `supplier_reliability`   | Numeric        | Supplier reliability score between 0 and 1      |
| `port_congestion`        | Numeric        | Port congestion indicator between 0 and 1       |
| `weather_severity`       | Numeric        | Weather severity indicator between 0 and 1      |
| `customs_delay_risk`     | Numeric        | Customs delay risk indicator between 0 and 1    |
| `inventory_days`         | Numeric        | Available inventory coverage in days            |
| `historical_delay_rate`  | Numeric        | Historical delay rate between 0 and 1           |
| `shipment_cost`          | Numeric        | Shipment cost                                   |
| `supplier_country`       | Categorical    | Supplier country                                |
| `transport_mode`         | Categorical    | Shipment transport mode                         |
| `product_category`       | Categorical    | Product or material category                    |
| `criticality`            | Categorical    | Operational criticality of the shipment         |
| `is_delayed`             | Binary integer | Known outcome: 1 for delayed, 0 for not delayed |

The manufacturer-specific workflow currently uses this schema independently of the general model's three additional corridor features.

### Target variable

The manufacturer-specific target column is:

```text
is_delayed
```

Its values must be:

- `1`: The shipment was delayed.
- `0`: The shipment was not delayed.

The target must represent a known outcome, not a prediction or estimate.

Only upload records whose shipment outcomes are known. A dataset without reliable outcome labels cannot support meaningful supervised evaluation.

### Data validation rules

The manufacturer upload workflow checks that:

- All required columns are present.
- Numeric features contain valid numeric values.
- Required feature values are not missing.
- The target contains only `0` and `1`.
- Reliability, congestion, weather, customs risk, and historical delay rate are between `0` and `1`.
- Order quantity and planned lead time are greater than zero.
- Inventory days and shipment cost are non-negative.
- Both delayed and non-delayed outcomes are represented.
- At least 50 records are available.
- Each outcome class contains at least 10 records.

These thresholds allow the application to reject some unsuitable datasets. They do not guarantee that a dataset is representative or large enough for reliable real-world predictions.

### Example manufacturer CSV header

```csv
order_quantity,planned_lead_time_days,supplier_reliability,port_congestion,weather_severity,customs_delay_risk,inventory_days,historical_delay_rate,shipment_cost,supplier_country,transport_mode,product_category,criticality,is_delayed
```

This is a header only. Each subsequent row must contain values in the corresponding columns.

---

## Prediction Explanations

AYEN's general prediction workflow includes a sensitivity-based explanation intended to help users understand which input features most affect an individual prediction.

The explanation process:

1. Calculates the original predicted delay probability.
2. Changes one input feature at a time to a reference value.
3. Calculates the new probability after each change.
4. Measures the difference from the original probability.
5. Ranks the features by the magnitude of that difference.
6. Displays the top three features as an explanation of the prediction.

Reference values are taken from the available reference data where supported by the implementation.

### How to interpret the explanation

A feature that causes a larger change in the model's output when modified receives a higher ranking.

For example, if changing a dwell-time feature substantially changes the predicted probability, that feature may appear among the top three reasons.

However:

- The explanation describes the model's sensitivity, not a proven real-world cause.
- Changing a feature to a reference value may create a combination of values that would not occur in reality.
- Correlated features can influence one another's apparent importance.
- A feature's ranking does not mean it independently caused a shipment delay.
- Explanations depend on the selected model and reference data.

Use the explanation as an aid for investigating a prediction, not as a substitute for operational evidence.

---

## Model Evaluation

AYEN displays standard classification metrics to help users inspect model performance.

### Accuracy

The proportion of evaluated records for which the model predicted the correct class.

Accuracy can be misleading when one outcome is much more common than the other.

### Precision

Among records predicted as delayed, the proportion that were actually delayed in the evaluation data.

Higher precision generally means fewer false-positive delay predictions.

### Recall

Among records that were actually delayed, the proportion the model correctly identified as delayed.

Higher recall means the model identified more of the delayed records in the evaluation data.

### F1 score

The harmonic mean of precision and recall. It provides a combined measure of these two metrics.

### ROC-AUC

Measures how well the model ranks delayed records above non-delayed records across possible classification thresholds.

A value near `0.5` generally indicates little ranking discrimination, while a value closer to `1.0` indicates stronger discrimination on the evaluated data.

### Evaluation procedure

The general training script uses an 80/20 train-test split.

The manufacturer-specific workflow uses a 75/25 train-test split.

Both workflows use stratified splitting to preserve approximate class proportions in the training and evaluation subsets.

Manufacturer-specific evaluation metrics are calculated using the held-out subset before the final model is retrained on the full dataset.

These are holdout evaluations, not comprehensive validation across different time periods or future shipments.

### Interpreting performance responsibly

The general model is trained on synthetic data. Strong performance on synthetic records would not, by itself, demonstrate that the model can accurately predict real shipment delays.

Likewise, a high accuracy score may simply reflect a dataset in which one outcome is much more common than the other.

Evaluation should consider:

- Accuracy, precision, recall, F1, and ROC-AUC together.
- The proportion of delayed and non-delayed records.
- Performance relative to a simple baseline.
- The quality and realism of the input features.
- Whether the model generalizes to unseen, real-world shipments.
- The business consequences of false-positive and false-negative predictions.

Real-world validation with verified shipment histories is required before relying on AYEN for operational decisions.

---

## Risk Interpretation

AYEN groups predicted delay probabilities into three application-defined risk categories.

| Predicted probability       | Risk category |
| --------------------------- | ------------- |
| Below `0.40`                | Low           |
| From `0.40` to below `0.70` | Medium        |
| `0.70` or above             | High          |

These thresholds are configurable application rules. They are not calibrated estimates of business loss or universal industry standards.

For example, a predicted probability of `0.75` means the model assigns a 75% predicted probability of delay based on its learned patterns. It does not mean the shipment is certain to be delayed.

Risk categories should be interpreted alongside supplier updates, current transport conditions, inventory availability, and other operational evidence.

The recommendations generated by AYEN should be reviewed by a person with relevant operational knowledge before action is taken.

---

## Data Storage

AYEN uses local filesystem storage in its current MVP implementation.

### General data

The general dataset is stored at:

```text
data/ayen_synthetic_shipments.csv
```

### General model artifacts

The general model and its evaluation metrics are stored at:

```text
models/disruption_model.joblib
models/model_metrics.json
```

### Manufacturer data

Manufacturer-specific data is stored under:

```text
data/manufacturers/
```

Each manufacturer directory can contain:

```text
history.csv
model.joblib
metrics.json
```

The history file contains uploaded shipment records. The model file contains the trained model. The metrics file contains the model's evaluation results.

Back up important data before replacing project files, cleaning directories, or changing the storage implementation.

The current filesystem storage is suitable for a local demonstration. It does not, by itself, provide a secure multi-tenant data architecture.

On hosting platforms such as Streamlit Community Cloud, local filesystem changes may not persist reliably across restarts or redeployments. A production deployment should use appropriate persistent storage and access controls.

---

## Configuration and Customization

### Change the general model

The general model training logic is in `train_model.py`.

Changes to the general model's feature list must be reflected consistently in the data generator, training pipeline, and prediction interface.

### Change manufacturer-specific models

Manufacturer-specific validation and training logic is in `manufacturer_models.py`.

Changes to the manufacturer schema must be reflected in the upload validation, training, and assessment workflows.

Keep the manufacturer-specific feature schema separate from the general model's corridor-related feature additions unless the manufacturer workflow is deliberately redesigned to support them.

### Change synthetic data

The general dataset generator is:

```text
generate_data.py
```

The manufacturer demonstration dataset generator is:

```text
generate_manufacturer_data.py
```

Generated data should remain clearly identified as synthetic.

### Change risk thresholds

The risk category thresholds are defined in `app.py`, in the `risk_band` function.

Changing these thresholds changes how probabilities are grouped into low, medium, and high risk. It does not retrain the model or improve the underlying predictions.

---

## Troubleshooting

### The general model is unavailable

Train the general model:

```bash
python train_model.py
```

If the dataset is missing, generate it first:

```bash
python generate_data.py
python train_model.py
```

Confirm that these files exist after training:

```text
models/disruption_model.joblib
models/model_metrics.json
```

### The general dataset is missing

Run:

```bash
python generate_data.py
```

Then train the model:

```bash
python train_model.py
```

### The application cannot find a trained manufacturer

Open Data Import, confirm that the manufacturer's history has been saved, and train the manufacturer's model.

A saved history alone does not mean a trained model exists.

### A CSV upload is rejected

Check the column names and values against the [Dataset Schemas](#dataset-schemas) section.

Common causes include:

- Missing columns.
- Missing feature values.
- Invalid risk scores.
- Unknown shipment outcomes.
- Too few records.
- Too few examples in one outcome class.

Remember that manufacturer-specific uploads use a different schema from the general model's dataset.

### A Python package is missing

Activate the correct virtual environment and run:

```bash
python -m pip install -r requirements.txt
```

Make sure the Python interpreter running Streamlit is the same interpreter where the dependencies were installed.

### The browser shows an application error

Check the terminal where Streamlit is running. The terminal usually contains the traceback or error message needed to identify the problem.

### Recently trained model changes are not reflected

Stop the Streamlit process with `Ctrl+C` and restart it:

```bash
streamlit run app.py
```

Restarting can help clear cached resources from the previous application process.

### Previously uploaded manufacturer data is missing

Check whether this directory still exists:

```text
data/manufacturers/
```

If the project was replaced or copied into a different directory, the application may now be using a different local storage location.

Restore data only from a known backup. Do not overwrite a newer history with an older backup unless that is intentional.

---

## Limitations and Security Considerations

AYEN is an MVP and has several important limitations.

### Synthetic training data

The included general and manufacturer demonstration datasets are synthetic. They are designed to demonstrate the software workflow, not to represent verified shipment conditions in Ethiopia or elsewhere.

The Ethiopian corridor-related features are simulated. They do not provide live visibility into port operations, dry port conditions, customs systems, or foreign exchange availability.

Metrics calculated on synthetic data do not establish real-world predictive performance.

### Data quality

Prediction quality depends on accurate, representative, and consistently recorded input data.

Incorrect labels, missing context, changes in logistics conditions, and differences between historical and future shipments can all reduce model reliability.

### Probability calibration

The model outputs a predicted probability, but the application has not established that its probabilities are calibrated against real-world shipment outcomes.

A displayed probability should not be treated as a guaranteed frequency of future delays.

### Prediction explanations

The top-three explanations are sensitivity-based. They describe how changing input features affects the model's output under the explanation method; they do not establish causal relationships.

### Local file storage

Manufacturer histories and trained models are stored on the filesystem. This approach does not provide authentication, authorization, encryption at rest, audit logging, or company-level access control by itself.

Do not use this MVP to store sensitive commercial shipment data for multiple companies in an untrusted shared environment.

### Deployment persistence

On hosting platforms such as Streamlit Community Cloud, local filesystem changes may not persist reliably across restarts or redeployments.

A production deployment should use appropriate persistent database or object storage and should be designed to protect company data.

### No live external intelligence integration

The current application does not independently verify predictions against live supplier feeds, port data, weather services, customs systems, or foreign exchange systems.

Users should verify current operational conditions through trusted sources.

### Human oversight

AYEN is intended to support decisions, not replace professional judgment. Operational recommendations should be reviewed against current business priorities, safety requirements, contractual commitments, and financial constraints.

---

## Future Development

Potential improvements include:

- Integration with reliable live port, weather, customs, and logistics data.
- Verification of Ethiopian corridor indicators against operational data.
- Secure authentication and role-based access control.
- Strong isolation between manufacturer accounts.
- Persistent database storage and backup management.
- Time-based model validation using future shipment records.
- Model calibration and decision-threshold analysis.
- Monitoring of prediction quality after deployment.
- More advanced prediction explanation methods.
- Data-quality reports and validation feedback.
- Automated model retraining with version tracking.
- Cost-sensitive evaluation based on the consequences of delayed and non-delayed predictions.
- Deployment automation and repeatable testing.

These are potential future directions, not claims that the current version already implements them.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

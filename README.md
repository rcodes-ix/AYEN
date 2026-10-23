# AYEN: Supply Chain Disruption Intelligence

AYEN is an AI-powered supply chain intelligence application designed to help manufacturers assess shipment disruption risk, explore historical shipment data, and make more informed logistics and inventory decisions.

The application combines machine learning-based delay prediction with shipment analytics and actionable recommendations. It supports a general prediction model as well as manufacturer-specific models trained on uploaded shipment histories.

AYEN is currently a minimum viable product (MVP) intended for demonstration, experimentation, and further development. Its included datasets are synthetic, and its predictions should not be treated as verified real-world forecasts.

## Table of Contents

* [Overview](#overview)
* [Key Features](#key-features)
* [How AYEN Works](#how-ayen-works)
* [Application Pages](#application-pages)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Requirements](#requirements)
* [Installation](#installation)
* [Running the Application](#running-the-application)
* [General Model Training](#general-model-training)
* [Manufacturer-Specific Models](#manufacturer-specific-models)
* [Using the Demonstration Dataset](#using-the-demonstration-dataset)
* [Dataset Schema](#dataset-schema)
* [Model Evaluation](#model-evaluation)
* [Risk Interpretation](#risk-interpretation)
* [Data Storage](#data-storage)
* [Configuration and Customization](#configuration-and-customization)
* [Troubleshooting](#troubleshooting)
* [Limitations and Security Considerations](#limitations-and-security-considerations)
* [Future Development](#future-development)
* [Contributing](#contributing)
* [License](#license)

## Overview

Supply chain disruptions can affect production schedules, inventory availability, transportation costs, and customer commitments. Manufacturers need to understand potential risks before those risks become operational problems.

AYEN provides a starting point for analyzing shipment-related risk using information such as:

* Supplier reliability
* Planned shipment lead time
* Transport mode
* Port congestion
* Weather severity
* Customs delay risk
* Inventory coverage
* Historical supplier delay rate
* Shipment cost
* Product criticality

The application uses these variables to estimate whether a shipment may be delayed. It also provides analytical views for exploring shipment records and understanding model evaluation results.

Two modeling workflows are supported:

1. **General AYEN model:** A baseline model trained on the general AYEN dataset.
2. **Manufacturer-specific model:** A separate model trained using a manufacturer's uploaded historical shipment records.

The manufacturer-specific workflow is intended to allow predictions to reflect patterns in a manufacturer's own data rather than relying exclusively on the general demonstration dataset.

## Key Features

### 1. Shipment disruption risk prediction

Users can enter shipment information and receive a model-generated estimate of delay risk.

The application presents the estimated probability alongside a risk category and recommendations based on the available shipment information.

### 2. General and manufacturer-specific models

Users can select between the general AYEN model and a trained manufacturer-specific model.

Manufacturer-specific models are trained independently using each manufacturer's saved historical dataset.

### 3. Manufacturer profiles and historical data

Users can create manufacturer profiles and upload historical shipment records.

The application validates the uploaded data before saving it for analysis and model training.

### 4. Shipment analytics

The application provides an overview of shipment data, including metrics and charts. Users can switch between the general dataset and available manufacturer histories.

### 5. Shipment exploration

Users can explore shipment records and examine patterns in the selected dataset.

### 6. Model transparency

The application displays available evaluation metrics for the selected model, helping users understand its measured performance on evaluation data.

### 7. Operational recommendations

Based on the predicted risk and shipment characteristics, AYEN can suggest actions involving supplier communication, inventory review, logistics monitoring, customs documentation, and production continuity.

These recommendations are decision-support suggestions, not guarantees that a particular action will prevent a disruption.

## How AYEN Works

The application follows this general workflow:

1. Load the general shipment dataset.
2. Load the trained general model and its evaluation metrics, if available.
3. Allow users to explore shipment records or enter information about a new shipment.
4. Select the general model or an available trained manufacturer-specific model.
5. Process shipment features using the preprocessing steps included in the selected model.
6. Generate a delay probability.
7. Display the prediction and an associated risk category.
8. Present relevant recommendations based on the prediction and shipment characteristics.

For manufacturer-specific training, the workflow is slightly different:

1. Create a manufacturer profile.
2. Upload historical shipment data.
3. Validate the uploaded records.
4. Save the validated history.
5. Split the records into training and evaluation subsets.
6. Train a model and evaluate it on the held-out subset.
7. Retrain the final model using all available records.
8. Save the model and evaluation metrics for future use.

The saved manufacturer-specific model can then be selected for new assessments.

## Application Pages

AYEN contains five primary pages.

### Overview

Provides summary statistics and charts for the selected dataset.

Users can choose the general AYEN dataset or a manufacturer's saved shipment history.

The displayed results depend on the selected data source.

### New Assessment

Provides a form for assessing the disruption risk of a shipment.

Users select the model source and, when applicable, the manufacturer whose trained model should be used.

The application returns a predicted delay probability, a risk category, and recommendations.

A prediction is an estimate based on the model's learned patterns. It is not confirmation that a shipment will or will not be delayed.

### Shipment Explorer

Provides an interface for exploring shipment records.

Users can select the general dataset or a manufacturer's saved history to investigate shipment characteristics and patterns.

### Data Import

Supports manufacturer profile creation, historical dataset upload, data validation, history storage, and manufacturer-specific model training.

A dataset must contain the required feature columns and known delay outcomes.

### Model Transparency

Displays evaluation metrics for the general model or a selected trained manufacturer-specific model.

This page is intended to make model evaluation results easier to inspect. Evaluation metrics should be interpreted alongside dataset quality, class balance, and the limitations of the evaluation process.

## Technology Stack

AYEN is built with the following technologies:

| Technology     | Purpose                                                                   |
| -------------- | ------------------------------------------------------------------------- |
| Python         | Application logic and model training                                      |
| Streamlit      | Interactive web application                                               |
| Pandas         | Dataset loading, validation, and analysis                                 |
| NumPy          | Numerical operations and synthetic data generation                        |
| Scikit-learn   | Machine learning pipelines, preprocessing, model training, and evaluation |
| Joblib         | Saving and loading trained models                                         |
| Plotly Express | Interactive charts                                                        |

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
│   └── manufacturer_demo.csv
│
├── models/
│   ├── disruption_model.joblib
│   └── model_metrics.json
│
└── data/
    └── manufacturers/
        └── <manufacturer_id>/
            ├── history.csv
            ├── model.joblib
            └── metrics.json
```

The `models/` directory is generated when the general model is trained. It may not exist in a fresh checkout.

The `data/manufacturers/` directory is created as manufacturer profiles and histories are saved. Its contents are generated at runtime and may not exist initially.

The directory tree above is illustrative. The application and scripts create generated directories when needed.

### File descriptions

**`app.py`**

Contains the Streamlit application, navigation, shipment assessment interface, dataset selection, visualizations, and model evaluation views.

**`manufacturer_models.py`**

Contains the functions used to validate manufacturer datasets, save and load shipment histories, train manufacturer-specific models, and load their evaluation metrics.

**`train_model.py`**

Trains and evaluates the general AYEN baseline model. Saves the trained model and its evaluation metrics under `models/`.

**`generate_data.py`**

Generates the general synthetic shipment dataset.

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

## Requirements

Before running AYEN, ensure that you have:

* Python 3.10 or a compatible Python version supported by the dependencies.
* `pip` installed.
* Access to a terminal.
* A working Python virtual environment.
* The project files and included dataset.

The project can run locally on Linux, WSL Ubuntu, or another compatible Python environment.

## Installation

### 1. Open the project directory

In a WSL terminal:

```bash
cd ~/AYEN
```

Confirm that the project files are present:

```bash
ls
```

You should see `app.py`, `train_model.py`, `manufacturer_models.py`, and `requirements.txt`.

### 2. Activate your existing virtual environment

If you already have a working virtual environment named `myenv`, activate it using the path where that environment is stored.

For example, if it is located in your home directory:

```bash
source ~/myenv/bin/activate
```

If the environment is already active, as indicated by its name in your terminal prompt, you do not need to activate it again.

Alternatively, create a new virtual environment inside the project:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Run:

```bash
python -m pip install -r requirements.txt
```

This installs the required packages into the active Python environment.

If installation fails, check the error message and confirm that the active Python version is compatible with the dependencies.

## Running the Application

Before starting the application for the first time, train the general model.

From the project directory, run:

```bash
python train_model.py
```

After training completes successfully, start Streamlit:

```bash
streamlit run app.py
```

Streamlit will display a local address in the terminal. Open that address in your browser.

To stop the application, return to the terminal and press `Ctrl+C`.

## General Model Training

The general model is trained using the dataset at:

```text
data/ayen_synthetic_shipments.csv
```

To train it:

```bash
python train_model.py
```

The script:

1. Loads the general dataset.
2. Checks that the required features and target column exist.
3. Separates the input features from the delay outcome.
4. Splits the data into training and testing subsets.
5. Builds preprocessing pipelines for numeric and categorical variables.
6. Trains a Random Forest classifier.
7. Calculates evaluation metrics on the held-out test subset.
8. Saves the trained pipeline and metrics.

The resulting files are:

```text
models/disruption_model.joblib
models/model_metrics.json
```

The model includes preprocessing steps, so categorical variables are encoded consistently during training and prediction.

If the general dataset is missing, the script instructs you to generate it first:

```bash
python generate_data.py
```

Then run:

```bash
python train_model.py
```

Retraining replaces the existing general model and its metrics.

## Manufacturer-Specific Models

AYEN supports training a separate model for each manufacturer using its saved shipment history.

### Step 1: Open Data Import

Start the application:

```bash
streamlit run app.py
```

Navigate to the Data Import page.

### Step 2: Create a manufacturer profile

Enter a manufacturer name and create its profile through the application interface.

Use a consistent name for each manufacturer so that its records remain associated with the intended profile.

### Step 3: Upload historical shipment records

Upload a CSV containing the required features and known shipment outcomes.

The application validates the data before saving it.

### Step 4: Save the history

Save the validated dataset to the manufacturer profile.

The history is stored locally under:

```text
data/manufacturers/<manufacturer_id>/history.csv
```

The actual directory name is derived from the manufacturer name.

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

* **New Assessment:** Select the manufacturer-specific model to assess a new shipment.
* **Overview:** Select the manufacturer's history to view its available analytics.
* **Shipment Explorer:** Explore the manufacturer's saved shipment records.
* **Model Transparency:** Inspect the manufacturer's model evaluation metrics.

Manufacturer-specific models are trained independently. Training one manufacturer's model does not automatically retrain the general model or other manufacturers' models.

## Using the Demonstration Dataset

The project includes:

```text
data/manufacturer_demo.csv
```

This file contains synthetic shipment records and can be used to test the manufacturer-specific workflow without needing to prepare a dataset first.

To use it:

1. Launch AYEN.
2. Open Data Import.
3. Create a test manufacturer profile.
4. Upload `data/manufacturer_demo.csv`.
5. Save the history.
6. Train the manufacturer's model.
7. Open New Assessment and select the trained manufacturer.
8. Inspect the results and open Model Transparency to review its metrics.

The demo dataset is fabricated for development and demonstration. It is not real manufacturer shipment history and must not be presented as evidence of real-world predictive performance.

To regenerate the manufacturer demo dataset, run:

```bash
python generate_manufacturer_data.py
```

This replaces the existing demonstration CSV with a newly generated synthetic dataset.

## Dataset Schema

Manufacturer uploads must contain the following 14 columns.

| Column                   | Type           | Description                                              |
| ------------------------ | -------------- | -------------------------------------------------------- |
| `order_quantity`         | Numeric        | Quantity ordered                                         |
| `planned_lead_time_days` | Numeric        | Planned shipment lead time in days                       |
| `supplier_reliability`   | Numeric        | Supplier reliability score between 0 and 1               |
| `port_congestion`        | Numeric        | Port congestion indicator between 0 and 1                |
| `weather_severity`       | Numeric        | Weather severity indicator between 0 and 1               |
| `customs_delay_risk`     | Numeric        | Customs delay risk indicator between 0 and 1             |
| `inventory_days`         | Numeric        | Available inventory coverage in days                     |
| `historical_delay_rate`  | Numeric        | Historical delay rate between 0 and 1                    |
| `shipment_cost`          | Numeric        | Shipment cost                                            |
| `supplier_country`       | Categorical    | Supplier country                                         |
| `transport_mode`         | Categorical    | Shipment transport mode                                  |
| `product_category`       | Categorical    | Product or material category                             |
| `criticality`            | Categorical    | Operational criticality of the shipment                  |
| `is_delayed`             | Binary integer | Known shipment outcome: 1 for delayed, 0 for not delayed |

### Target variable

The target column is:

```text
is_delayed
```

Its values must be:

* `1`: The shipment was delayed.
* `0`: The shipment was not delayed.

The target must represent a known outcome, not a prediction or an estimate.

Only upload records whose shipment outcomes are known. A dataset without reliable outcome labels cannot support meaningful supervised evaluation.

### Data validation rules

The manufacturer upload workflow checks that:

* All required columns are present.
* Numeric features contain valid numeric values.
* Required feature values are not missing.
* The target contains only 0 and 1.
* Reliability, congestion, weather, customs risk, and historical delay rate are between 0 and 1.
* Order quantity and planned lead time are greater than zero.
* Inventory days and shipment cost are non-negative.
* Both delayed and non-delayed outcomes are represented.
* At least 50 records are available.
* Each outcome class contains at least 10 records.

These minimum thresholds allow the application to reject some unsuitable datasets. They do not guarantee that a dataset is sufficiently representative or large enough for reliable real-world predictions.

### Example CSV header

```csv
order_quantity,planned_lead_time_days,supplier_reliability,port_congestion,weather_severity,customs_delay_risk,inventory_days,historical_delay_rate,shipment_cost,supplier_country,transport_mode,product_category,criticality,is_delayed
```

The example is a header only. Each subsequent row must contain values in the corresponding columns.

## Model Evaluation

AYEN displays several standard classification metrics.

### Accuracy

The proportion of evaluated records for which the model predicted the correct class.

Accuracy can be misleading when one outcome is much more common than the other.

### Precision

Among records predicted as delayed, the proportion that were actually delayed in the evaluation data.

Higher precision means fewer false positive delay predictions, although the result depends on the selected decision threshold and evaluation dataset.

### Recall

Among records that were actually delayed, the proportion the model correctly identified as delayed.

Higher recall means the model identified more of the delayed records in the evaluation data.

### F1 score

The harmonic mean of precision and recall. It provides a combined measure of these two metrics.

### ROC-AUC

Measures how well the model ranks delayed records above non-delayed records across possible classification thresholds.

A value near 0.5 generally indicates little ranking discrimination, while a value closer to 1.0 indicates stronger discrimination on the evaluated data.

### Evaluation procedure

For the general model, the training script uses an 80/20 train-test split.

For manufacturer-specific models, the application uses a 75/25 train-test split.

Both workflows use stratified splitting to preserve the approximate class proportions in the training and evaluation subsets.

Manufacturer-specific evaluation metrics are calculated using the held-out subset before the final model is retrained on the full dataset.

These are single holdout evaluations, not comprehensive validation across different time periods or future shipments.

Metrics should be interpreted together with the dataset size, label quality, class balance, and the real-world context in which the model will be used.

## Risk Interpretation

The application groups predicted delay probabilities into three risk categories:

| Predicted probability   | Risk category |
| ----------------------- | ------------- |
| Below 0.40              | Low           |
| From 0.40 to below 0.70 | Medium        |
| 0.70 or above           | High          |

These categories are application-defined thresholds, not calibrated estimates of business loss or universal industry standards.

For example, a predicted probability of 0.75 means the model estimates a 75% probability of delay under its learned patterns. It does not mean the shipment is certain to be delayed.

A risk category should be interpreted alongside supplier updates, current transport conditions, inventory availability, and other operational evidence.

The application also generates recommendations based on the predicted risk and shipment characteristics. These recommendations should be reviewed by a person with relevant operational knowledge before action is taken.

## Data Storage

AYEN uses local filesystem storage in its current MVP implementation.

### General data

The general dataset is stored at:

```text
data/ayen_synthetic_shipments.csv
```

### General model artifacts

The general model and metrics are stored at:

```text
models/disruption_model.joblib
models/model_metrics.json
```

### Manufacturer data

Each manufacturer has a separate local directory under:

```text
data/manufacturers/
```

A manufacturer directory can contain:

```text
history.csv
model.joblib
metrics.json
```

The history file contains uploaded shipment records. The model file contains the trained model. The metrics file contains the model's evaluation results.

These files are important application data. Back them up before replacing project files, cleaning directories, or changing the storage implementation.

The current filesystem storage is suitable for a local demonstration. It is not, by itself, a secure multi-tenant data architecture.

## Configuration and Customization

### Change the model

The general model training logic is in `train_model.py`.

Manufacturer-specific model validation and training logic is in `manufacturer_models.py`.

Changes to model features, preprocessing, or training parameters should be applied consistently wherever those features are used, including the assessment interface and uploaded-data validation.

### Change the synthetic data

The general dataset generator is `generate_data.py`.

The manufacturer demonstration dataset generator is `generate_manufacturer_data.py`.

Generated data should remain clearly identified as synthetic.

### Change risk thresholds

The risk category thresholds are defined in `app.py`, in the `risk_band` function.

Changing these thresholds changes how predicted probabilities are grouped into low, medium, and high risk. It does not retrain the model or improve the quality of its underlying predictions.

## Troubleshooting

### The application says the general model is unavailable

Train the general model:

```bash
python train_model.py
```

Confirm that these files exist afterward:

```text
models/disruption_model.joblib
models/model_metrics.json
```

If training fails, inspect the error output and verify that the general dataset exists and contains the required columns.

### The general dataset is missing

Generate it:

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

Check the column names and values against the [Dataset Schema](#dataset-schema).

Common causes include missing columns, missing feature values, invalid risk scores, unknown shipment outcomes, or too few examples in one outcome class.

### A Python package is missing

Activate the correct virtual environment and run:

```bash
python -m pip install -r requirements.txt
```

Make sure the Python interpreter running Streamlit is the same interpreter where the dependencies were installed.

### The browser shows an application error

Check the terminal where Streamlit is running. The terminal usually contains the traceback or error message needed to identify the problem.

### Previously uploaded manufacturer data is missing

Check whether the directory below still exists:

```text
data/manufacturers/
```

If the project was replaced or copied into a different directory, the application may now be looking at a different local storage location.

Restore data only from a known backup. Do not overwrite a newer history with an older backup unless that is intentional.

## Limitations and Security Considerations

AYEN is an MVP and has several important limitations.

### Synthetic training data

The included general and manufacturer demonstration datasets are synthetic. They are designed to demonstrate the software workflow, not to represent verified shipment conditions in Ethiopia or elsewhere.

Metrics calculated on synthetic data do not establish real-world predictive performance.

### Data quality

Prediction quality depends on accurate, representative, and consistently recorded input data.

Incorrect labels, missing context, changes in logistics conditions, and differences between historical and future shipments can all reduce model reliability.

### No guarantee of calibrated probabilities

The model's output is a predicted probability. The application has not established that those probabilities are calibrated against real-world outcomes.

A displayed probability should not be treated as a guaranteed frequency of future delays.

### Local file storage

Manufacturer histories and trained models are stored on the filesystem. This approach does not provide authentication, authorization, encryption at rest, audit logging, or company-level access control by itself.

Do not use this MVP to store sensitive commercial shipment data for multiple companies in an untrusted shared environment.

### Deployment persistence

On hosting platforms such as Streamlit Community Cloud, local filesystem changes may not persist reliably across restarts or redeployments.

A production deployment should use an appropriate persistent database or object storage service and should be designed to protect company data.

### No live external intelligence integration

The current application does not independently verify predictions against live supplier feeds, port data, weather services, customs systems, or other real-time sources.

Users should verify current operational conditions through trusted sources.

### Human oversight

AYEN is intended to support decisions, not replace professional judgment. Operational recommendations should be reviewed against current business priorities, safety requirements, contractual commitments, and financial constraints.

## Future Development

Potential improvements include:

* Integration with reliable live weather, port, and logistics data.
* Secure authentication and role-based access control.
* Strong isolation between manufacturer accounts.
* Persistent database storage and backup management.
* Time-based model validation using future shipment records.
* Model calibration and decision-threshold analysis.
* Monitoring of prediction quality after deployment.
* More detailed explanations of individual predictions.
* Data-quality reports and validation feedback.
* Automated model retraining with version tracking.
* Cost-sensitive evaluation based on the consequences of delayed and non-delayed predictions.
* Deployment automation and repeatable testing.

These are potential future directions, not claims that the current version already implements them.

## Contributing

Contributions that improve reliability, usability, data validation, testing, documentation, or model evaluation are welcome.

Before submitting changes:

1. Create a separate branch for your work.
2. Keep changes focused and document important design decisions.
3. Test modified Python files for syntax errors.
4. Test the affected application workflows.
5. Do not commit confidential manufacturer histories or sensitive data.
6. Do not present synthetic data or synthetic-data metrics as real-world evidence.

## License

No license is specified in this repository by this README.

Before distributing or reusing the project, add a `LICENSE` file that explicitly states the terms under which the code may be used, modified, and redistributed.

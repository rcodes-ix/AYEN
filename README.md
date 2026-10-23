# AYEN baseline project

This is a demo MVP using reproducible synthetic shipment data. It is not trained on real Ethiopian logistics records and its metrics are not evidence of real-world predictive performance.

## Setup
1. Create/activate a Python virtual environment.
2. Install: `pip install -r requirements.txt`
3. Generate data: `python generate_data.py`
4. Train/evaluate: `python train_model.py`
5. Launch the app after adding `app.py`: `streamlit run app.py`

`is_delayed` is 1 when actual lead time exceeds planned lead time, otherwise 0.
The model only uses fields intended to be known before the shipment outcome.

6. Launch app: `streamlit run app.py`

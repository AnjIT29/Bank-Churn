{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "88767763-98ba-4ee6-b7c2-fbabe1ce35a3",
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: 'model.pkl'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 7\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mpandas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mpd\u001b[39;00m\n\u001b[0;32m      6\u001b[0m \u001b[38;5;66;03m# Load model and scaler\u001b[39;00m\n\u001b[1;32m----> 7\u001b[0m model \u001b[38;5;241m=\u001b[39m joblib\u001b[38;5;241m.\u001b[39mload(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmodel.pkl\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m      8\u001b[0m scaler \u001b[38;5;241m=\u001b[39m joblib\u001b[38;5;241m.\u001b[39mload(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mscaler.pkl\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m     10\u001b[0m st\u001b[38;5;241m.\u001b[39mtitle(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m💳 Bank Customer Churn Prediction System\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\joblib\\numpy_pickle.py:650\u001b[0m, in \u001b[0;36mload\u001b[1;34m(filename, mmap_mode)\u001b[0m\n\u001b[0;32m    648\u001b[0m         obj \u001b[38;5;241m=\u001b[39m _unpickle(fobj)\n\u001b[0;32m    649\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m--> 650\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28mopen\u001b[39m(filename, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mrb\u001b[39m\u001b[38;5;124m'\u001b[39m) \u001b[38;5;28;01mas\u001b[39;00m f:\n\u001b[0;32m    651\u001b[0m         \u001b[38;5;28;01mwith\u001b[39;00m _read_fileobject(f, filename, mmap_mode) \u001b[38;5;28;01mas\u001b[39;00m fobj:\n\u001b[0;32m    652\u001b[0m             \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(fobj, \u001b[38;5;28mstr\u001b[39m):\n\u001b[0;32m    653\u001b[0m                 \u001b[38;5;66;03m# if the returned file object is a string, this means we\u001b[39;00m\n\u001b[0;32m    654\u001b[0m                 \u001b[38;5;66;03m# try to load a pickle file generated with an version of\u001b[39;00m\n\u001b[0;32m    655\u001b[0m                 \u001b[38;5;66;03m# Joblib so we load it with joblib compatibility function.\u001b[39;00m\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: 'model.pkl'"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import joblib\n",
    "import pandas as pd\n",
    "\n",
    "# Load model and scaler\n",
    "model = joblib.load(\"model.pkl\")\n",
    "scaler = joblib.load(\"scaler.pkl\")\n",
    "\n",
    "st.title(\"💳 Bank Customer Churn Prediction System\")\n",
    "\n",
    "st.write(\"Fill the details below to predict whether the customer will exit.\")\n",
    "\n",
    "# User Input Fields (Same order as training features)\n",
    "input_data = []\n",
    "\n",
    "columns = model.n_features_in_\n",
    "feature_names = pd.read_csv(\"friance new.csv\").drop(\"Exited\", axis=1).columns\n",
    "\n",
    "for feature in feature_names:\n",
    "    value = st.number_input(f\"Enter value for {feature}\", value=0.0)\n",
    "    input_data.append(value)\n",
    "\n",
    "# Prediction Button\n",
    "if st.button(\"Predict\"):\n",
    "    user_array = np.array([input_data])\n",
    "    user_scaled = scaler.transform(user_array)\n",
    "    prediction = model.predict(user_scaled)[0]\n",
    "\n",
    "    if prediction == 1:\n",
    "        st.error(\"🔴 The customer is likely to **EXIT / CHURN** 🔻\")\n",
    "    else:\n",
    "        st.success(\"🟢 The customer will **STAY / NOT CHURN** 👍\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8eea5bc4-bb04-4197-8d23-77399612df78",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

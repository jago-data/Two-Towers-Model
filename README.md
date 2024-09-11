---
# Two-Towers-Recommendation Model

This repository contains code for generating synthetic data, training a Two-Tower recommendation model, and predicting top product recommendations for users. The recommendations are filtered to include only the top 5 per product type for each user.

## Project Structure

1. **Data Generation**
   - Generates synthetic user data, product data, and interaction data.
   - Encodes categorical columns and creates negative samples.

2. **Model Building and Training**
   - Builds a Two-Tower model using TensorFlow/Keras.
   - Trains the model with the generated data.

3. **Prediction and Filtering**
   - Predicts recommendations for all user-product combinations in chunks.
   - Filters and returns the top 5 recommendations per product type for each user.

## Dependencies

- `polars` - For data manipulation.
- `numpy` - For numerical operations.
- `pandas` - For data manipulation and conversion.
- `scikit-learn` - For data encoding and splitting.
- `tensorflow` - For building and training the recommendation model.

## Installation

You can install the necessary packages using pip:

```bash
pip install polars numpy pandas scikit-learn tensorflow
```

## Usage

1. **Generate Synthetic Data**

   The script generates synthetic data for users, products, and interactions. It also creates negative samples for training purposes.

2. **Train the Model**

   The model is built using TensorFlow/Keras and trained on the synthetic data. The Two-Tower model architecture is used to predict user preferences for products.

3. **Predict and Filter Recommendations**

   Predictions are made in chunks to handle large-scale data efficiently. The results are filtered to include the top 5 recommendations per product type for each user.

## Code Details

### Data Generation

The code generates:
- `user_data` with customer IDs, age groups, segments, income, and AUM.
- `interaction_data` with customer-product interactions.
- `product_data` with product codes, names, types, and performances.

### Model Building

The Two-Tower model consists of:
- **User Tower**: Embeds user features.
- **Product Tower**: Embeds product features.
- **Dot Product**: Computes similarity between user and product embeddings.

### Prediction

Predictions are made in chunks to manage memory efficiently:
- Inputs are prepared and predictions are generated for each chunk.
- Predictions are filtered to keep the top 5 per product type for each user.

### Saving Results

The final top recommendations are saved to a DataFrame, which can be converted to a CSV file if needed.

## Example Output

The code prints the top recommendations DataFrame:

```python
# Print the first few rows of the DataFrame with top recommendations
print(top_recommendations_pl_df)
```

## Notes

- Ensure sufficient memory capacity when adjusting the `chunk_size`.
- The model's performance and accuracy depend on the quality of the generated data and the number of epochs during training.

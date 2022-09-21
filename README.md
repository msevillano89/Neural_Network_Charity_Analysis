# Neural_Network_Charity_Analysis
## Overview of the analysis
This analysis aims to create a binary classifier capable of predicting whether applicants will be successful if funded by Alphabet Soup. They accomplish this task, Alphabet Soup's business team has provided a dataset with information from 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are several columns that capture metadata about each organization, such as the following:

- EIN and NAME: Identification columns
- APPLICATION_TYPE: Alphabet Soup application type
- AFFILIATION: Affiliated sector of industry
- CLASSIFICATION: Government organization classification
- USE_CASE: Use case for funding
- ORGANIZATION: Organization type
- STATUS: Active status
- INCOME_AMT: Income classification
- SPECIAL_CONSIDERATIONS: Special consideration for application
- ASK_AMT: Funding amount requested
- IS_SUCCESSFUL: Was the money used effectively

The project will consist of three deliverables:
- **Deliverable 1**: Preprocessing Data for a Neural Network Model
- **Deliverable 2**: Compile, Train, and Evaluate the Model
- **Deliverable 3**: Optimize the Model

## Results
### Data Preprocessing
**1. What variable(s) are considered the target(s) for your model?**

The target variable for the model is the 'IS_SUCCESSFUL' column. The column is compromised by binary values, where the value 1 indicates that the applicant used the money effectively, and 0 means that it was unsuccessful.

**2. What variable(s) are considered to be the features for your model?**

Features for this model are the  'STATUS', 'ASK_AMT', 'IS_SUCCESSFUL', 'APPLICATION_TYPE', 'CLASSIFICATION', 'USE_CASE', 'ORGANIZATION', 'INCOME_AMT'. I dropped "USE_CASE_Other","AFFILIATION_Other" columns.

**3. What variable(s) are neither targets nor features, and should be removed from the input data?**

We will remove the 'EIN' and 'NAME' columns as they are not beneficial to the model's accuracy.

### Compiling, Training, and Evaluating the Model
**1. How many neurons, layers, and activation functions did you select for your neural network model, and why?**

I opted for Relu and Sigmoid Activations Functions since Sigmoid is best used for models with binary classification and Relu for nonlinear datasets. In addition, I decided to have a high number of neurons to ascertain if I could achieve high accuracy at the initial stages of building the model. The initial setup is described below.
```
# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train[0])
hidden_nodes_layer1 = 80
hidden_nodes_layer2 = 30

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
)

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()
```
```
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_6 (Dense)             (None, 80)                3520      
                                                                 
 dense_7 (Dense)             (None, 30)                2430      
                                                                 
 dense_8 (Dense)             (None, 1)                 31        
                                                                 
=================================================================
Total params: 5,981
Trainable params: 5,981
Non-trainable params: 0
```
The above setup resulted in an accuracy of **%72.64**.

**2. Were you able to achieve the target model performance?**

No, I could not achieve the target model performance of 75%. I try to improve the accuracy by adding additional hidden layers and changing the number of neurons on each layer. However, the model's accuracy did not rise above 72%.

**3. What steps did you take to try and increase model performance?**

- Reviewed data frame to identify other non-beneficial columns. I opted to leave columns as they were.
- Since I had started with layers with a high number of neurons. I opted to add a hidden layer and decrease the number of neurons in my original layers. 
- Finally, I experimented with changing the number of neurons in each layer, increasing it with each iteration. 

## Summary
The highest accuracy the model was able to achieve was 72.64% accuracy after various optimization iterations. We could consider using a different model, such as a Random Forest Classifier, and compare the model's accuracy to the one we originally developed. The Random Forest Classifier might be a good option as they tend to be less influenced by outliers. 

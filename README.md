#Project Summary
The goal of this project is to predict the interest rate of a loan, based off various payment and income information. This dataset includes all loan data for all loans issued by Lending Club from 2007 to 2019, comprising over 2 million observations. Due to computing constraints, the analysis to the first 10,000 observations. With this data, I will explore different standard regression and machine learning models, as well as their sensitivity to the different test-train data splits.

#Specifications
Used in this project are standard regression models, polynomial regression, regression trees, random forest modeling, generalized additive models (GAMs), boosting, and neural networks. Many of these have additional parameters within the model, and these values were chosen thorugh cross-validation, as seen in the full report. As well, I used training splits of around 5%, 10%, and 50% of the data. These represent the small, medium, and large seen below.

#Results
Model | Size | Test Error
------------ | -------------
Multiple Regression | Small | 19.99 
 | Medium | 19.06
 | Large | 19.05
 | Small | 662.92
 | Medium | 280.64
 | Large | 106.41
  | Small | 21.89
 | Medium | 21.23
 | Large | 20.59
  | Small | 20.86
 | Medium | 19.37
 | Large | 27.35
  | Small | 20.24
 | Medium | 19.05
 | Large | 19.04
  | Small | 19.20
 | Medium | 18.78
 | Large | 18.40
  | Small | 22.82
 | Medium | 23.05
 | Large | 22.48



#Conclusion
With all of our models, we had a range of test errors from to 18.7 to over 250. Overall, the GAM and boosting models performed the best across the three sizes of our training data in the c. The models with less training data tended to have higher test errors, and were especially high when using models with large variance, such as polynomials. Models with higher interpretability and a more visual form, such as regular trees and neural networks had test error rates of around 10-20% higher than more standard models, but are likely easier to explain. The test errror rates on average declined as the size. Overall, the varying results in this project for different models, tuning parameters, and training data sizes show the importance of trying multiple approaches to analysis in order to have comprehensive analysis. Please see Full Report for more specifics.

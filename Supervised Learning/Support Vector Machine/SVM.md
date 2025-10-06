## Support Vector Machine

useful for high dimensional spaces (also when dimensions are greater than number of samples only if you appropriately chose kernel functions and regularization functions critically)

Uses subset of trainingpoints in decision functions which are called the support vectors
Different kernel functions can be used on decision functions.

They dont give simple probabiltiies across the space but are usign five fold cross validation

SVC have three typesm the standard, NuSVC and Linear SVC
SVC is used for binary or multiclass classification

There are different types of kernels that dictate the linearlity of the line created
Linear kernel RBF Kernel and n-polinomial degree kernel

Linear svg uses squared hinge loss

they have parameters of:
intercept scaling
decision_function_shape

Set modified weights in case of class imbalance
sample_weight -> C * class_weight[i]


Support Vector Regression: Used for regression problems
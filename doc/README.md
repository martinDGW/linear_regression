```
F(x) = mx + b
```

- y (F(x)) is the closest value for x
- x is the fix value on our dataset
- m is the slope (pente) of the line determining whether an increase of km (x) corresponds to an increase or decrease in the price (y)
- b (intercept) represents the value of y (the price) when x is zero. It’s the vertical offset of the line from the origin.

---
- On one side, there’s the Mean Squared Error (MSE), the cost function indicating how far we deviate from the correct price, concerning the reference km B, also known as the residual.

- On the other side, there’s gradient descent, an algorithm striving to minimize the MSE. It adjusts parameters b (the base price) and m (the slope) in each study session, based on the gradient and moves in the opposite direction. The gradient indicates where the error is leading us if uncorrected, guiding us to move precisely in the opposite direction.

These adjustments rely on partial derivatives, indicating how the cost function changes concerning variations in the model parameters (such as slope and intercept). These derivatives guide the algorithm to iteratively update the parameters, including those associated with external factors, to minimize the overall error and reach a point of minimum cost.

For example, if J(m,b)* is the cost function (MSE), the partial derivatives used in the gradient descent would be:
```
∂J/∂m​ (Partial derivative with respect to m)
∂J/∂b (Partial derivative with respect to b)
```

These partial derivatives indicate how the cost function changes concerning variations in m and b, allowing the algorithm to iteratively update these parameters to reach a minimum point (minimum cost).

They also depend on our learning rate (α), representing the correction quantity we aim for in every session.

# LONG SHORT
---
The fundemantal Equation: y = mx + b

The formula represents the linear relationship between note frequencies (y) and note positions on the staff (x). m is the slope indicating how price changes with kms, and b is the value adjustment needed to reduce error.

---
MSE (Mean Squared Error)

Measures the deviation between reel and desired price. It quantifies how much the price deviates from the desired value.

---
Gradient Descent

This method gradually updates price value in each iteration during practice. It uses partial derivatives to correct price, reducing the error concerning desired value.

---
Partial Derivatives

Partial derivatives indicate how a price changes with respect to model parameters.

In linear regression and Gradient Descent, these partial derivatives of the cost function concerning model parameters (in this case b and m) help understand how the function’s slope changes in each direction. Specifically, ∂/∂m represents the cost function’s change concerning m. Gradient Descent utilizes these partial derivatives to iteratively update parameter values in the opposite direction of the gradient. The update process follows the gradient rule:

New parameter value = Old parameter value − α × Corresponding partial derivative

Where α is the learning rate determining the step size during optimization. This iterative process continues until convergence towards optimal parameter values.

---
Learning Rate (α)

Represents the correction to make in each iteration during practice. A higher α indicates more substantial corrections, while a lower α indicates finer corrections. A higher learning rate might imply a propensity for sudden or radical changes in the execution method. While beneficial in some instances, it might also risk instability or difficulties in maintaining consistency in improving a specific method. Conversely, a lower learning rate might indicate a preference for making corrections more gradually and thoughtfully. This approach might be more suitable when aiming to refine existing methods or subtler details without disrupting the entire approach. The choice of the learning rate, is often a balancing act. It must be calibrated to foster constant improvement without compromising stability and consistency in the learning or execution process.

---
Iterations

Represent a practice sessions, during which corrections are made to their performances. Iterations continue until the price gradually approaches the desired value.

---
Residual

Represents the residual error between the price finded by algorithm and the desired price. Indicates how much the program can still improve to approach the desired value.

# Calculate m
m = covariance (X; Y) / variance(X)
  = moyenne(xy) - moyenne(x) * moyenne(y) / moyenne(x(carre)) - carre(moyenne(x))

# Calculate b
b = moyenne(y) - a * moyenne(x)

pour ces caculs on a besoin de 5 colonnes:
X | Y | XY | X carre | Y carre

somme
moyenne

# Coefficient de correlation lineaire
r = covariance(X; Y) / ecart-type(X) * ecart-type(Y)
r = cov(X;Y) / racine(variance(x)) * racine(variance(y))
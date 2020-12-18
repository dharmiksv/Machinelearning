# Machinelearning

Logistic Regression is a statstical method for predicting binary classes or classifying data into discrete outcomes.

binary classification problem: y can take on only two values, 0 and 1. 
Hence, y∈{0,1}. 0 is also called the negative class, and 1 the positive class
Assume x^{(i)} is the features of the PNR data set, and y^{(i)} may be '1' if trip is a Business Travel and '0' if trip is a Leisure.

Hypothesis Representation:

Sigmoid Function or Logistic Function:

hθ(x)=g(θ^T*x)   -- hθ(x) will give us probability that the trip is a Business Travel i.e '1' and it is '0' is just the complement of our probability that it is 1. 
                    hθ(x) =P(y=1|x;θ)=1−P(y=0|x;θ) (Bernoulli distribution)
                    P(y=0|x;θ)+P(y=1|x;θ)=1
z=θ^T*x
g(z)=1/(1+e^(−z))

The function g(z), shown here, maps any real number to the (0, 1) interval.









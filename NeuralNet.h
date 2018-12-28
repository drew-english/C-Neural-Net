#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

using std::vector;
using std::endl;
using std::cerr;

class Network
{
public:
  // constructs a new network based on parameters
  Network(int inputs, int hiddenLayers, int numHidden,
    int outputs, double(*actHidden)(double x), double(*actOut)(double x));
  ~Network(); // destructor for a network

  vector<double> run(vector<double> const &input); //computes the given output of the network from the input
  void fit(vector<double> const &input, vector<double> const &target); //Updates the weights of the nework based on the paramters

private:
  int inputs, hiddenLayers, numHidden, outputs, totalWeights; // total # of each value
  vector<double> cache, hiddenNeurons, weights; // stores real values of each
  double(*actFunOut)(double x); // neuron activation function for the output
  double(*actFunHidden)(double x); // neuron activation function of the hidden layers
};

//acivation functions and their derivatives
//relu gives values [0, infinity] while sigmoid gives [0, 1]
//sigmoid is usually better for probabilities while relu is usually better for real values
//linear for range of real valued outputs
double sigmoid(double x);
double d_sigmoid(double x);
double relu(double x);
double d_relu(double x);
double unrelu(double x);
double linear(double x);
double d_linear(double x);

#endif // NerualNet.h

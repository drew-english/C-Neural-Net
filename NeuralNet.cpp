#include "NeuralNet.h"

//Activation funcitons:
double sigmoid(double x)
{ return 1 / (1 + exp(-x)); }

double d_sigmoid(double x)
{ return sigmoid(x) * (1 - sigmoid(x)); }

double relu(double x) // softplus relu
{ return log10(1 + exp(x)); }

double d_relu(double x)
{ return exp(x) / (1 + exp(x)); }

double unrelu(double x)
{	return log(pow(10, x) - 1); }

double linear(double x)
{  return x; }

double d_linear(double x)
{	return 1; }


Network::Network(int inputs, int hiddenLayers, int numHidden,
  int outputs, double(*actHidden)(double x), double(*actOut)(double x))
{
  //checks to make sure fucntion arguments are valid
  try
  {
	  if (inputs <= 0)
      throw("Invalid number of inputs to network");
    if (hiddenLayers < 0)
      throw("Invalid number of hidden layers");
	  if (hiddenLayers > 0 && numHidden <= 0)
      throw("Invalid number of hidden neurons");
	  if (outputs <= 0)
      throw("Invalid number of outputs");
  }
  catch(const char * msg)
  { std::cerr << "Error During Network Construction: " << msg << '\n'; }

  //calculating totalWeights,
  //adding 1 to neuron layers for a bias in each layer
	if (hiddenLayers != 0)
    this->totalWeights = ((inputs + 1) * numHidden) + ((hiddenLayers - 1) * ((numHidden + 1) * numHidden)) +
  	((numHidden + 1) * outputs);
	else
  	this->totalWeights = (inputs + 1) * outputs;

  this->inputs = inputs;
  this->hiddenLayers = hiddenLayers;
  this->numHidden = numHidden;
  this->totalWeights = totalWeights;
  this->outputs = outputs;
  this->actFunHidden = actHidden;
  this->actFunOut = actOut;

  srand(time(NULL)); // initializes random seed
	for (int i = 0; i < this->totalWeights; i++) // randomly setting weights and initializes cache to all 0
	{
		this->weights.push_back(1 * ((double)rand() / (double)RAND_MAX));
		if (rand() % 2)
			this->weights[i] *= -1;

    if(i < this->numHidden)
      this->hiddenNeurons.push_back(0);

		this->cache.push_back(0);
	}
}

//no memory is dynamically allocated, besides when inserting into the vector containers
// which is then taken care of by their class.
Network::~Network(){};

vector<double> Network::run(vector<double> const &input)
{
  //makes sure input is the correct size
  try
  {
    if((int)input.size() != this->inputs)
      throw("Invalid input size");
  }
  catch(const char *msg)
  { std::cerr << "Error While Running Network: " << msg << endl;  }


  double sum; // stores running sum for each neuron (before activation function)
  vector<double>::iterator w = this->weights.begin();
  vector<double>::iterator neur = this->hiddenNeurons.begin();

  for(int i = 0; i < this->hiddenLayers; i++) // each hidden layer
  {
    for(int j = 0; j < this->numHidden; j++) // each neruon in hidden layer
    {
      sum = *w++; // sum starts with bias

      for(int k = 0; k < (i == 0 ? this->inputs : this->numHidden); k++) // each neruon in previous layer
      {
        sum += *w++ * (i == 0 ? input[k] : this->hiddenNeurons[k]); // sum += current weight multiplied by previous neuron value
      }

      *neur++ = this->actFunHidden(sum); // current neuron value is updated
    }
  }

  vector<double> output; // output layer that is returned by this function
  for(int i = 0; i < this->outputs; i++) // each output
  {
    sum = *w++; // sum starts with bias

    for(int j = 0; j < (this->hiddenLayers > 0 ? this->numHidden : this->inputs); j++) // each neuron in previous layer
    {
      sum += *w++ * (this->hiddenLayers > 0 ? //sum += current weight multiplied by previous neuron value
        this->hiddenNeurons[j + ((this->hiddenLayers - 1) * this->numHidden)] : input[j]);
    }

    output.push_back(this->actFunOut(sum)); // current neuron value is updated
  }

  return output;
}

void Network::fit(vector<double> const &input, vector<double> const &target)
{
  try // checking for errors in input or target sizes
  {
    if((int)input.size() != this->inputs)
      throw("Invalid input size");
    if((int)target.size() != this->outputs)
      throw("Invalid target size");
  }
  catch(const char *msg)
  { std::cerr << "Error While Fitting Network: " << msg << endl;  }

  
}

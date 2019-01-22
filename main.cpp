#include "NeuralNet.h"

using std::cout;

int main(int argc, char *argv[])
{
  Network net(20, 1, 10, 5, relu, linear);
  vector<double> input, output, tOut;
  
  //create input and target output
  for(int i = 0; i < 20; i++)
  {
    if(i < 5)
      tOut.push_back(5);
    
    input.push_back(4);
  }
  
  //run network and see initial output
  output = net.run(input);
  cout << "Initial:" << endl;
  for(int i = 0; i < 5; i++)
    cout << output[i] << endl;
  cout << endl;

  //fit network with data then print output after
  for(int i = 0; i < 500; i++)
    net.fit(input, tOut);
  output = net.run(input);
  for (int i = 0; i < 5; i++)
    cout << output[i] << endl;

  //save network to file for later use
  net.save("Saves/NetworkSave.data");
  
  //loads a new network based on net
  Network net2("Saves/NetworkSave.data");

  //runs net2 to test if loaded correctly
  output = net2.run(input);
  cout << endl; 
  for(int i = 0; i < 5; i++)
    cout << output[i] << endl;
  return 0;
}

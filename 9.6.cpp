#include<iostream>
#include<cmath>
#include<vector>
#include<Eigen/Dense>
#include<Eigen/QR>
using namespace std;
using namespace Eigen;

void increLogi( MatrixXd &trainData, VectorXd &label, double &phi_0, 
				VectorXd &phi, VectorXd &eps )
{
	phi_0 = 0;
	VectorXd activation(trainData.cols());

	
}
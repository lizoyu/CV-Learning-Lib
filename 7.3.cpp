/*************************************************************************
	> File Name: 7.3.cpp - Maximum Likelihood for factor analyzer
	> Author: 
	> Mail: 
	> Created Time: 2017年02月26日 星期日 09时36分31秒
 ************************************************************************/

#include<iostream>
#include<cmath>
#include<vector>
#include<Eigen/Dense>
#include<Eigen/QR>
#include<Eigen/Eigenvalues>
using namespace std;
using namespace Eigen;

void MLforFA( MatrixXd &trainData, int num_factor, VectorXd &mean, 
			MatrixXd &factors, MatrixXd &var )
{
	\\ trainData: (I, D); mean: (trainData.rows(), 1); factors: (rows(), factors);
	\\ var: (rows(), rows()).
	\\ initialize
	mean = trainData.rowwise().mean();
	factors = MatrixXd::Random(factors.rows(),factors.cols()).abs();
	for( int i = 0; i < var.rows(); ++i )
	{
		VectorXd deviation = trainData.row(i) - mean(i);
		deviation = deviation.array() * deviation.array();
		var(i,i) = deviation.sum();
	}
	var = var / trainData.cols();

	\\ E-step

	\\ M-step
	
}

void inferFA( VectorXd &mean, MatrixXd &factors, MatrixXd &var, double prior)
{

}

int main()
{

	return 0;
}
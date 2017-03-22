/*************************************************************************
	> File Name: 9.2.cpp - Bayesian logistic
	> Author: 
	> Mail: 
	> Created Time: 2017年03月04日 星期六 09时16分03秒
 ************************************************************************/

#include<iostream>
#include<cmath>
#include<vector>
#include<Eigen/Dense>
#include<Eigen/QR>
#include"CVLearning.h"
using namespace std;
using namespace Eigen;

void BayeforLogi( MatrixXd &trainData, VectorXd &label, double var_init, 
				VectorXd &mean, MatrixXd &var )
{
	// optimize to get parameter phi
	VectorXd phi;
	phi = MAPforLogi( trainData, label, var_init );

	// compute 2nd order derivative
	MatrixXd H;
	H = MatrixXd::Ones(trainData.rows,trainData.rows()).array() / var_init;
	for( size_t i = 0; i < trainData.cols(); ++i )
	{
		double y = 1 / (1 + exp(-phi.transpose()*trainData.col(i)) );
		H += y*(1-y)*trainData.col(i)*trainData.col(i).transpose();
	}

	// set mean and covariance
	mean = phi;
	CompleteOrthogonalDecomposition<MatrixXd> cod( H );
	var = -cod.pseudoInverse();
}

VectorXd InferBayeLogi( VectorXd &mean, MatrixXd &var, VectorXd point )
{
	double mean_a;
	mean_a = mean.transpose() * point;
	double var_a;
	var_a = point.transpose() * var * point.transpose();

	// approximate intergral for Bernoulli parameter lambda
	double lambda;
	lambda = 1 / (1 + exp(-mean_a / sqrt(1 + M_PI*var_a/8)));

	// predictive distribution
	VectorXd l(2);
	l(0) = 1 - lambda;
	l(1) = lambda;

	return l;
}

int main()
{
	// two class, 1 - (10, 20, 30); 0 - (70, 80, 90);
    MatrixXd trainData(3,6);
    trainData << 1.3, 0.9, 1.1, 7.4, 6.8, 7.1,
                 2.2, 1.8, 2.1, 8.2, 7.6, 8.1,
                 3.1, 2.4, 3.1, 9.1, 8.9, 9.1;
    trainData *= 10;
    VectorXd label(6);
    label << 1, 1, 1, 0, 0, 0;
    VectorXd point(3);
    point << 68, 78, 88;

    // learn
    double var_init = 100;
    VectorXd mean;
    MatrixXd var;
    BayeforLogi( trainData, label, 100, mean, var );

    // predict
    VectorXd l;
    l = InferBayeLogi( mean, var, point );
    cout << "Likelihood:" << endl << l << endl;
	return 0;
}
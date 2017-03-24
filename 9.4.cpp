#include<iostream>
#include<cmath>
#include<vector>
#include<Eigen/Dense>
#include<Eigen/QR>
using namespace std;
using namespace Eigen;

void BayeforDualLogi( MatrixXd &trainData, VectorXd &label, double var_init, 
				VectorXd &mean, MatrixXd &var )
{
	// optimize to get parameter phi
	//initialize
	MatrixXd data = MatrixXd::Ones(trainData.rows()+1,trainData.cols());
	data.block(1, 0, trainData.rows(), trainData.cols()) = trainData;
	double L = -data.cols()*log10(2*M_PI*var_init)/2;
	VectorXd g;
	MatrixXd H;

	// Newton iteration(cost minimized)
	double L_old = L - 1;
	VectorXd psi(data.cols());
	while( L - L_old > 0.001 )
	{
		// compute prediction, L, g, H
		double y;
		L_old = L;
		for( int i = 0; i < label.size(); ++i )
		{
			y = 1/(1+exp(-psi.transpose()*data.transpose()*data.col(i)));
			if( label(i) == 1 && y != 0 )
				L -= log10(y);
			else if( y!= 1 )
				L -= log10(1-y);
			g += (y - label(i)) * data.transpose() * data.col(i);
			H += y*(1-y)*data.transpose()*data.col(i)*data.col(i).transpose()*data;
		}

		// compute new estimate
		CompleteOrthogonalDecomposition<MatrixXd> cod( H );
		psi -= cod.pseudoInverse() * g;
	}

	// set mean and covariance
	mean = psi;
	CompleteOrthogonalDecomposition<MatrixXd> cod( H );
	var = -cod.pseudoInverse();
}

VectorXd InferBayeLogi( VectorXd &mean, MatrixXd &var, MatrixXd &trainData, 
						VectorXd point )
{
	VectorXd data_p = VectorXd::Ones(point.size()+1);
	data_p.tail(point.size()) = point;
	MatrixXd data = MatrixXd::Ones(trainData.rows()+1,trainData.cols());
	data.block(1, 0, trainData.rows(), trainData.cols()) = trainData;
	double mean_a;
	// start here
	mean_a = mean.transpose() * data;
	double var_a;
	var_a = data.transpose() * var * data;

    cout << "mean_a:" << mean_a << endl;
    cout << "var_a:" << var_a << endl;

	// approximate intergral for Bernoulli parameter lambda
	double lambda;
	lambda = 1 / (1 + exp(-mean_a / sqrt(1 + M_PI*var_a/8)));

	// predictive distribution
	VectorXd l(2);
	l(0) = lambda;
	l(1) = 1 - lambda;

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
    BayeforLogi( trainData, label, var_init, mean, var );
    cout << "mean:" << endl << mean << endl;
    cout << "var:" << endl << var << endl;

    // predict
    VectorXd l;
    l = InferBayeLogi( mean, var, point );
    cout << "Likelihood:" << endl << l << endl;
	return 0;
}
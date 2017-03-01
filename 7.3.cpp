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
	// trainData: (I, D); mean: (trainData.rows(), 1);
	// factors: (rows(), num_factor); var: (rows(), rows()).
	// initialize
	mean = trainData.rowwise().mean();
	factors = MatrixXd::Random(factors.rows(),factors.cols()).array().abs();
	for( int i = 0; i < var.rows(); ++i )
	{
		VectorXd deviation = trainData.col(i) - mean;
		deviation = deviation.array() * deviation.array();
		var(i,i) = deviation.sum();
	}
	var = var / trainData.cols();
	cout << "mean: " << endl << mean << endl;
	cout << "factors:" << endl << factors << endl;
	cout << "var:" << endl << var << endl;
	double L, L_old = 1;
	while( L - L_old != 0 )
	{
		// E-step
		// E_h: (num_factor, cols()); E_hh: (num_factor, num_factor*num_factor).
		MatrixXd E_h(num_factor,trainData.cols());
		MatrixXd E_hh(trainData.cols()*num_factor,num_factor);
		CompleteOrthogonalDecomposition<MatrixXd> cod( var );
		for( int i = 0; i < trainData.cols(); ++i )
		{
			VectorXd deviation = trainData.col(i) - mean;
			MatrixXd temp = factors.transpose()*cod.pseudoInverse()*factors + 
				MatrixXd::Ones(num_factor,num_factor);
			CompleteOrthogonalDecomposition<MatrixXd> cod_temp( temp );
			E_h.col(i) = cod_temp.pseudoInverse()*factors.transpose()*
				cod.pseudoInverse()*deviation;
			E_hh.block(i*num_factor, 0, num_factor, num_factor) = 
				cod_temp.pseudoInverse() + E_h.col(i)* E_h.col(i).transpose();
		}

		// M-step
		MatrixXd first(trainData.rows(),num_factor), second(num_factor,num_factor);
		first.setZero(); second.setZero();
		for( int i = 0; i < trainData.cols(); ++i )
		{
			first += (trainData.col(i)-mean)*E_h.col(i).transpose();
			second += E_hh.block(i*num_factor, 0, num_factor, num_factor);
		}
		CompleteOrthogonalDecomposition<MatrixXd> cod_sec( second );
		factors = first*cod_sec.pseudoInverse();
		MatrixXd var_norm = var + factors*factors.transpose();
		if( var_norm.diagonal().minCoeff() > pow( 10, -13 ) )
		{
			var.setZero();
			for( int i = 0; i < trainData.cols(); ++i )
			{
				VectorXd deviation = trainData.col(i) - mean;
				var += deviation*deviation.transpose() - factors*E_h.col(i)*
					deviation.transpose();
			}
			VectorXd vec_diag = var.diagonal();
			MatrixXd mat_diag = vec_diag.asDiagonal();
			var = mat_diag / trainData.cols();
		}
		cout << "factors:" << endl << factors << endl;
		cout << "var:" << endl << var << endl;

		// Log Likelihood
		L_old = L;
		L = 0;
		MatrixXd var_norm = var + factors*factors.transpose();
		CompleteOrthogonalDecomposition<MatrixXd> cod_norm( var_norm );
		SelfAdjointEigenSolver<MatrixXd> es( var_norm );
		for( int i = 0; i < trainData.cols(); ++i )
		{
			VectorXd deviation = trainData.col(i) - mean;
			L += log10(exp(-0.5*deviation.transpose()*cod_norm.pseudoInverse()*
				deviation)/(pow((2*M_PI), round(trainData.rows()/2))*
				sqrt(abs(es.eigenvalues().sum()))));
		}
		cout << "L: " << L << endl;
	}	
}

double inferFA( VectorXd &mean, MatrixXd &factors, MatrixXd &var, 
				VectorXd point,double prior)
{
	double l;
	MatrixXd var_norm = var + factors*factors.transpose();
	CompleteOrthogonalDecomposition<MatrixXd> cod_norm( var_norm );
	SelfAdjointEigenSolver<MatrixXd> es( var_norm );
    VectorXd deviation = point - mean;
    l = exp(-0.5*deviation.transpose()*cod_norm.pseudoInverse()*
				deviation)/(pow((2*M_PI), round(point.size()/2))*
				sqrt(abs(es.eigenvalues().sum())));
	return l;
}

int main()
{
	// two class, 1 - (10, 20, 30); 2 - (70, 80, 90);
    MatrixXd trainData_1(3,3), trainData_2(3,3);
    trainData_1 << 1.3, 0.9, 1.1,
                   2.2, 1.8, 2.1,
                   3.1, 2.4, 3.1;
    trainData_2 << 7.4, 6.8, 7.1,
                   8.2, 7.6, 8.1,
                   9.1, 8.9, 9.1;
    trainData_1 *= 10;
    trainData_2 *= 10;
    VectorXd point(3);
    point << 78, 88, 98;

    // train two t-distribution model for each class
    int K = 2;
    VectorXd mean_1(trainData_1.rows()), mean_2(trainData_2.rows());
    MatrixXd var_1(mean_1.rows(),mean_1.rows()), var_2(mean_2.rows(),mean_2.rows());
    MatrixXd factors_1(mean_1.rows(),K), factors_2(mean_2.rows(),K);
    double prior_1 = 0.5, prior_2 = 0.5;
    MLforFA( trainData_1, K, mean_1, factors_1, var_1 );
    MLforFA( trainData_2, K, mean_2, factors_2, var_2 );
    // inference
/*
	double l_1 = inferFA( mean_1, factors_1, var_1, point, prior_1 );
    double l_2 = inferFA( mean_2, factors_2, var_2, point, prior_2 );
    vector<double> P(2);
    P[0] = l_1 / (l_1 + l_2);
    P[1] = l_2 / (l_1 + l_2);

    cout << "Probility: " << P[0] << " " << P[1] << endl;*/
	return 0;
}
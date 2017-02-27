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
	\\ trainData: (I, D); mean: (trainData.rows(), 1);
	\\ factors: (rows(), num_factor); var: (rows(), rows()).
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

	double L, L_old = 1;
	while( L - L_old != 0 )
	{
		\\ E-step
		\\ E_h: (num_factor, cols()); E_hh: (num_factor, num_factor*num_factor).
		MatrixXd E_h(num_factor,trainData.cols());
		MatrixXd E_hh(num_factor,num_factor*num_factor);
		CompleteOrthogonalDecomposition<MatrixXd> cod( var );
		for( int i = 0; i < trainData.cols(); ++i )
		{
			VectorXd deviation = trainData.col(i) - mean;
			MatrixXd temp = factors.transpose()*cod.pseudoInverse()*factors + 
				MatrixXd::Ones(num_factor,num_factor);
			CompleteOrthogonalDecomposition<MatrixXd> cod_temp( var );
			E_h.col(i) = cod_temp.pseudoInverse()*factors.transpose()*
				cod.pseudoInverse()*deviation;
			E_hh.block(i*num_factor, 0, num_factor, num_factor) = 
				cod_temp.pseudoInverse() + E_h.col(i)* E_h.col(i).transpose();
		}

		\\ M-step
		MatrixXd first(trainData.rows(),num_factor), second(num_factor,num_factor);
		first.setZero(); second.setZero();
		for( int i = 0; i < trainData.cols(); ++i )
		{
			first += (trainData.col(i)-mean)*E_h.col(i).transpose();
			second += E_hh.block(i*num_factor, 0, num_factor, num_factor);
		}
		CompleteOrthogonalDecomposition<MatrixXd> cod_sec( second );
		factors = first*cod_sec.pseudoInverse();
		
	}	
}

void inferFA( VectorXd &mean, MatrixXd &factors, MatrixXd &var, double prior)
{

}

int main()
{

	return 0;
}
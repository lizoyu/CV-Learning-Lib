#include<iostream>
#include<cmath>
#include<vector>
#include<Eigen/Dense>
#include<Eigen/QR>
using namespace std;
using namespace Eigen;

double RVC( MatrixXd &trainData, VectorXd &label, double deg_free )
{
	MatrixXd HV = MatrixXd::Identity(trainData.cols(),trainData.cols());
	while(){
		// MAP estimate of psi
		// initialize
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
				VectorXd kernel(trainData.cols());
				for( int j = 0; j < kernel.size(); ++j )
				{
					VectorXd minus = trainData.col(j) - trainData.col(i);
					kernel(j) = exp(-0.5*(minus.transpose()*minus/1));
				}
				y = 1/(1+exp(-psi.transpose()*kernel));
				if( label(i) == 1 && y != 0 )
					L -= log10(y);
				else if( y!= 1 )
					L -= log10(1-y);
				g += (y - label(i)) * kernel;
				H += y*(1-y)*kernel*data.col(i).transpose()*data;
			}

			// compute new estimate
			CompleteOrthogonalDecomposition<MatrixXd> cod( H );
			psi -= cod.pseudoInverse() * g;
		}
	}
}
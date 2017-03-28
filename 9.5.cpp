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
	MatrixXd HV_old = HV.array()*0.1;
	while( abs(HV.diagonal().sum() - HV_old.diagonal().sum()) > 0.1 ){
		HV_old = HV;
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
				H += y*(1-y)*kernel*kernel.transpose();
			}

			// compute new estimate
			CompleteOrthogonalDecomposition<MatrixXd> cod( H );
			psi -= cod.pseudoInverse() * g;
		}

		// compute Hessian S
		MatrixXd S = HV;
		for( int i = 0 ; i < trainData.cols(); ++i )
		{
			VectorXd kernel(trainData.cols());
			for( int j = 0; j < kernel.size(); ++j )
			{
				VectorXd minus = trainData.col(j) - trainData.col(i);
				kernel(j) = exp(-0.5*(minus.transpose()*minus/1));
			}
			double y = 1/(1+exp(-psi.transpose()*kernel));
			S += y*(1-y)*kernel*kernel.transpose();
		}

		// Set mean and variance
		VectorXd mean = psi;
		CompleteOrthogonalDecomposition<MatrixXd> cod( S );
		MatrixXd var = -cod.pseudoInverse();

		// update diagonal of H(hidden variables)
		for( int i = 0; i < trainData.cols(); ++i )
			HV(i,i) = ( 1 - HV(i,i)*var(i,i) + deg_free ) / 
				( pow(mean(i),2) + deg_free );
	}
	
	// remove related parameters according to hidden variables
	VectorXd tracker = VectorXd::LinSpaced(HV.cols(),0,HV.cols()-1);
	for( int i = 0; i < HV.cols(); ++i )
	{
		if( HV(i,i) > 100 )
		{
			double k = 0;
			while( tracker[k] != i ) k++;

		}
	}

}
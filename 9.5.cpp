#include<iostream>
#include<cmath>
#include<vector>
#include<Eigen/Dense>
#include<Eigen/QR>
using namespace std;
using namespace Eigen;

MatrixXd RVC( MatrixXd &trainData, VectorXd &label, double deg_free,
		VectorXd &mean, MatrixXd &var )
{
	MatrixXd data = MatrixXd::Ones(trainData.rows()+1,trainData.cols());
	data.block(1, 0, trainData.rows(), trainData.cols()) = trainData;
	MatrixXd HV = MatrixXd::Identity(data.cols(),data.cols());
	MatrixXd HV_old = HV.array()*0.1;
	while( abs(HV.diagonal().sum() - HV_old.diagonal().sum()) > 0.1 ){
		HV_old = HV;
		// MAP estimate of psi
		// initialize
		double L = -data.cols()*log10(2*M_PI*100)/2;
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
				VectorXd kernel(data.cols());
				for( int j = 0; j < kernel.size(); ++j )
				{
					VectorXd minus = data.col(j) - data.col(i);
					kernel(j) = exp(-0.5*minus.transpose()*minus);
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
		for( int i = 0 ; i < data.cols(); ++i )
		{
			VectorXd kernel(data.cols());
			for( int j = 0; j < kernel.size(); ++j )
			{
				VectorXd minus = data.col(j) - data.col(i);
				kernel(j) = exp(-0.5*minus.transpose()*minus);
			}
			double y = 1/(1+exp(-psi.transpose()*kernel));
			S += y*(1-y)*kernel*kernel.transpose();
		}

		// Set mean and variance
		mean = psi;
		CompleteOrthogonalDecomposition<MatrixXd> cod( S );
		var = -cod.pseudoInverse();

		// update diagonal of H(hidden variables)
		for( int i = 0; i < data.cols(); ++i )
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
			tracker.segment(k,tracker.size()-1-k) = tracker.tail(tracker.size()-1-k);
			tracker.conservativeResize(tracker.size()-1);
			mean.segment(k,mean.size()-1-k) = mean.tail(mean.size()-1-k);
			mean.conservativeResize(mean.size()-1);
			data.block(0,k,data.rows(),data.cols()-1-k) = 
				data.rightCols(data.cols()-1-k);
			data.conservativeResize(data.rows(),data.cols()-1);
			var.block(0,k,var.rows(),var.cols()-1-k) = 
				var.rightCols(var.cols()-1-k);
			var.conservativeResize(var.rows(),var.cols()-1);
			var.block(k,0,var.rows()-1-k,var.cols()) = 
				var.bottomRows(var.rows()-1-k);
			var.conservativeResize(var.rows()-1,var.cols());
		}
	}
	return data;
}

VectorXd inferRVC( VectorXd &mean, MatrixXd &var, MatrixXd &trainData,
					VectorXd &point )
{
	VectorXd point_p = VectorXd::Ones(point.size()+1);
	point_p.tail(point.size()) = point;
	VectorXd kernel(trainData.cols());
	for( int j = 0; j < kernel.size(); ++j )
	{
		VectorXd minus = trainData.col(j) - point_p;
		kernel(j) = exp(-0.5*minus.transpose()*minus);
	}
	double mean_a = mean.transpose()*kernel;
	double var_a = kernel.transpose()*var*kernel;
	double lambda = 1 / (1+exp(-mean_a/sqrt(1+M_PI*var_a/8)));
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
    point << 18, 28, 38;

    // learn
    double deg_free = 3;
    VectorXd mean;
    MatrixXd var, data;
    data = RVC( trainData, label, deg_free, mean, var );
    cout << "mean:" << endl << mean << endl;
    cout << "var:" << endl << var << endl;
    cout << "data:" << endl << data << endl;

    // predict
    VectorXd l;
    l = inferRVC( mean, var, data, point );
    cout << "Likelihood:" << endl << l << endl;
	return 0;
}
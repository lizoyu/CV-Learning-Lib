/*************************************************************************
	> File Name: 9.1.cpp - MAP logistic
	> Author: 
	> Mail: 
	> Created Time: 2017年03月04日 星期六 09时16分03秒
 ************************************************************************/

#include<iostream>
#include<cmath>
#include<vector>
#include<Eigen/Dense>
#include<Eigen/QR>
using namespace std;
using namespace Eigen;

VectorXd MAPforLogi( MatrixXd &trainData, VectorXd &label, double var_init )
{
	// trainData: (rows(),cols()); label: (cols(),1); phi: (rows()+1,1)
	// initialize
	MatrixXd data = MatrixXd::Ones(trainData.rows()+1,trainData.cols());
	data.block(1, 0, trainData.rows(), trainData.cols()) = trainData;
	double L = data.rows()*log10(2*M_PI*var_init)/2;
	VectorXd g;
	MatrixXd H;

	// Newton iteration(cost minimized)
	double L_old = L - 1;
	VectorXd phi(trainData.rows()+1);
	while( L - L_old > 0.001 )
	{
		// compute prediction, L, g, H
		double y;
		L_old = L;
		for( int i = 0; i < label.size(); ++i )
		{
			y = 1 / (1 + exp(-phi.transpose()*data.col(i)));
			if( label(i) == 1 && y != 0 )
				L -= log10(y);
			else if( y!= 1 )
				L -= log10(1-y);
			g += (y - label(i)) * data.col(i);
			H += y*(1-y)*data.col(i)*data.col(i).transpose();
		}

		// compute new estimate
		CompleteOrthogonalDecomposition<MatrixXd> cod( H );
		phi -= cod.pseudoInverse() * g;
	}
	return phi;
}

double InferMAPLogi( VectorXd &phi, VectorXd &point )
{
	VectorXd data = VectorXd::Ones(point.size()+1);
	data.tail(point.size()) = point;
	return 1 / (1 + exp(-phi.transpose()*data));
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

    // learn the parameter phi
    double var_init = 100;
    VectorXd phi;
    phi = MAPforLogi( trainData, label, var_init );
    cout << "Phi:" << endl << phi << endl;

    double l;
    l = InferMAPLogi( phi, point );
    cout << "Likelihood: " << l << endl;
	return 0;
}
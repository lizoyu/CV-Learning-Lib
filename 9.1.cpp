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

VectorXd MAPforLogi( MatrixXd &trainData, VectorXd &label, VectorXd &phi_init,
					 double var_init )
{
	// trainData: (rows(),cols()); label: (cols(),1); phi: (rows()+1,1)
	// initialize
	MatrixXd data = MatrixXd::Ones(trainData.rows()+1,trainData.cols());
	data.block(1, 0, trainData.rows(), trainData.cols()) = trainData;
	cout << "Data:" << endl << data << endl;
	double L = data.rows()*log10(2*M_PI*var_init)/2 +
				(double) (phi_init.transpose()*phi_init)/var_init;
	VectorXd g = phi_init.array() / var_init;
	MatrixXd H = MatrixXd::Ones(g.size(),g.size()).array() / var_init;

	// start here
	// Newton iteration(cost minimized)
	double L_old = L - 1;
	VectorXd phi = phi_init;
	while( L - L_old > 0.001 )
	{
		// compute new estimate
		CompleteOrthogonalDecomposition<MatrixXd> cod( H );
		phi = phi + cod.pseudoInverse()*g;
		L = data.rows()*log10(2*M_PI*var_init)/2 +
				(double)(phi.transpose()*phi)/var_init;
		g = phi.array() / var_init;
		H = MatrixXd::Ones(g.size(),g.size()).array() / var_init;

		// compute prediction, L, g, H
		double y;
		L_old = L;
		for( int i = 0; i < label.size(); ++i )
		{
			y = 1 / (1 + exp(-phi.transpose()*data.col(i)));
			cout << "y: " << y << endl;
			cout << "product: " << -phi.transpose()*data.col(i) << endl;
			if( label(i) == 1 && y != 0 )
				L -= log10(y);
			else if( y!= 1 )
				L -= log10(1-y);
			cout << "L: " << L << endl;
			g += (y - label(i)) * data.col(i);
			H += y*(1-y)*data.col(i)*data.col(i).transpose();
		}
		cout << "H:" << endl << H << endl;
		cout << "g:" << endl << g << endl;
		cout << "L: " << L << endl;
	}
	return phi;
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

    // learn the parameter phi
    VectorXd phi;
    VectorXd phi_init = VectorXd::Ones(trainData.rows()+1);
    double var_init = 100;
    phi = MAPforLogi( trainData, label, phi_init, var_init );

    cout << "Phi:" << endl << phi << endl;
	return 0;
}
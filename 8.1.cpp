/*************************************************************************
	> File Name: 8.1.cpp - ML learning for linear regression
	> Author: 
	> Mail: 
	> Created Time: 2017年03月05日 星期日 11时12分24秒
 ************************************************************************/

#include<iostream>
#include<cmath>
#include<vector>
#include<Eigen/Dense>
#include<Eigen/QR>
using namespace std;
using namespace Eigen;

void MLforLinear( MatrixXd &trainData, VectorXd &label, MatrixXd &phi, 
				  double &var )
{
	// trainData: (rows(),cols()); label: (cols(),1); phi: (rows(),1)
	data = MatrixXd::Ones(trainData.rows()+1,trainData.cols());
	data.block(1, 0, trainData.rows(), trainData.cols()) = trainData;
	l = VectorXd::Ones(label.size()+1);
	l.tail(label.size()) = label;

	// gradient
	MatrixXd temp_1 = data*data.transpose();
	CompleteOrthogonalDecomposition<MatrixXd> cod( temp_1 );
	phi = cod.pseudoInverse()*data*l;

	// variance
	MatrixXd temp_2 = l - data.transpose()*phi;
	var = temp_2.transpose() * temp_2 / trainData.cols();
}

void inferLinear( MatrixXd &phi, double var, VectorXd &point )
{

}

int main()
{
	// two class, 1 - (10, 20, 30); 2 - (70, 80, 90);
    int num_class = 2;
    MatrixXd trainData(3,6);
    trainData << 1.3, 0.9, 1.1, 7.4, 6.8, 7.1,
                 2.2, 1.8, 2.1, 8.2, 7.6, 8.1,
                 3.1, 2.4, 3.1, 9.1, 8.9, 9.1;
    trainData *= 10;
    VectorXd state(6);
    state << 1, 1, 1, 2, 2, 2;
    VectorXd point(3);
    point << 68, 78, 88;
}
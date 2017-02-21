/*************************************************************************
	> File Name: 7.2.cpp
	> Author: 
	> Mail: 
	> Created Time: 2017年02月19日 星期日 09时36分31秒
 ************************************************************************/

#include<iostream>
#include<cmath>
#include<vector>
#include<Eigen/Dense>
#include<Eigen/QR>
#include<Eigen/Eigenvalues>
#include<boost/math/special_functions/digamma.hpp>
using namespace std;
using namespace Eigen;
using namespace boost::math;

void MLforT( MatrixXd &trainData, VectorXd &mean, MatrixXd &var, double nu )
{
    // trainData; mean: (trainData.row(), 1); var: (row(), row())
    // initialize
    nu = 1000;
    mean = trainData.rowwise().mean();
    for( int i = 0; i < trainData.cols(); ++i )
    {
        VectorXd deviation = trainData.col(i) - mean;
        for( int j = 0; j < trainData.rows(); ++j )
            var(j,j) += pow( deviation(j), 2 );
    }
    
    double L, L_old = 1;
    while( L - L_old > 0.001 )
    {
        // E-step
        VectorXd delta(trainData.cols()), E(delta.size()), E_log(E.size());
    
        {CompleteOrthogonalDecomposition<MatrixXd> cod( var );
        for( int i = 0; i < trainData.cols(); ++i )
        {
            VectorXd deviation = trainData.col(i) - mean;
            delta(i) = deviation.transpose() * cod.pseudoInverse() * deviation;
            E(i) = ( nu + trainData.rows() ) / ( nu + delta(i) );
            E_log(i) = digamma( nu/2 + trainData.rows()/2 ) 
                - log10( nu/2 + delta(i)/2 );
        }}

        // M-step
        mean.setZero();
        for( int i = 0; i < trainData.cols(); ++i )
            mean += trainData.col(i) * E(i);
        mean = mean / E.sum();
        if( var.diagonal().minCoeff() > pow( 10, -13 ) )
        {
            var.setZero();
            for( int i = 0; i < trainData.cols(); ++i )
            {
                VectorXd deviation = trainData.col(i) - mean;
                var += E(i) * deviation * deviation.transpose();
            }
            VectorXd var_vec = var.diagonal();
            MatrixXd var_diag = var_vec.asDiagonal();
            var = var_diag / E.sum();
        }
        // compute degree of freedom

        // compute log likelihood
        {CompleteOrthogonalDecomposition<MatrixXd> cod( var );
        for( int i = 0; i < trainData.cols(); ++i )
        {
            VectorXd deviation = trainData.col(i) - mean;
            delta(i) = deviation.transpose() * cod.pseudoInverse() * deviation;
        }}
        L_old = L;
        {SelfAdjointEigenSolver<MatrixXd> es( var );       
        L = trainData.cols()*(log10(tgamma((nu+trainData.rows())/2)) - 
            trainData.rows()*log10(nu*M_PI)/2 - 
            log10(es.eigenvalues().sum())/2 - log10(tgamma(nu/2)));
        }
        double sum_log;
        for( i = 0; i < delta.size(); ++i )
            sum_log += log10( 1 + delta(i)/nu );
        L = L - ( nu + trainData.rows() ) * sum_log / 2;
    }
}

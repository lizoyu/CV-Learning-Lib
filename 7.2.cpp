/*************************************************************************
	> File Name: 7.2.cpp - Maximum Likelihood for t-distribution
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
#include<boost/math/special_functions/trigamma.hpp>
using namespace std;
using namespace Eigen;
using namespace boost::math;

void MLforT( MatrixXd &trainData, VectorXd &mean, MatrixXd &var, double &nu )
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
    double L = 1, L_old = 0;
    while( abs(L - L_old) > 0.001 )
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
        double t_cost_1, t_cost_2, t_cost, t_cost_old, err = 1;
        while( err != 0 )
        {
            t_cost_1 = E.sum()/2 - log10(nu/2)/2 - 1/log(10) - 
                digamma(nu/2)/log(10);
            t_cost_2 = - 1/log(10) - trigamma(nu/2)/log(10);
            t_cost_old = -nu*log10(nu/2)/2 - log10(tgamma(log(nu/2))) + 
                (nu/2 - 1)*E_log.sum() - nu*E.sum()/2;
            nu = nu - t_cost_1/t_cost_2;
            t_cost = -nu*log10(nu/2)/2 - log10(tgamma(log(nu/2))) + 
                (nu/2 - 1)*E_log.sum() - nu*E.sum()/2;
            err = abs((t_cost - t_cost_old) / t_cost);
        }

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
        double sum_log = 0;
        for( int i = 0; i < delta.size(); ++i )
            sum_log += log10( 1 + delta(i)/nu );
        L = L - ( nu + trainData.rows() ) * sum_log / 2;
    }
}

double inferT( VectorXd &mean, MatrixXd &var, double nu, VectorXd point,
           double prior )
{
    int D = mean.size();
    // compute likelihood for new point
    double l;
    CompleteOrthogonalDecomposition<MatrixXd> cod( var );
    SelfAdjointEigenSolver<MatrixXd> es( var );
    VectorXd deviation = point - mean;
    l = tgamma((nu+D))/2 * pow(1+(double)(deviation.transpose()*
        cod.pseudoInverse()*deviation)/nu, -(nu+D)/2) / 
        (pow(nu*M_PI, D/2)*pow(es.eigenvalues().sum(),0.5)*tgamma(nu/2));
    l *= prior;

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
    VectorXd mean_1(trainData_1.rows()), mean_2(trainData_2.rows());
    MatrixXd var_1(mean_1.rows(),mean_1.rows()), var_2(mean_2.rows(),mean_2.rows());
    double prior_1 = 0.5, prior_2 = 0.5, nu_1 = 1000, nu_2 = 1000;
    MLforT( trainData_1, mean_1, var_1, nu_1 );
    MLforT( trainData_2, mean_2, var_2, nu_2 );
    // inference
    double l_1 = inferT( mean_1, var_1, nu_1, point, prior_1 );
    double l_2 = inferT( mean_2, var_2, nu_2, point, prior_2 );
    vector<double> P(2);
    P[0] = l_1 / (l_1 + l_2);
    P[1] = l_2 / (l_1 + l_2);

    cout << "Probility: " << P[0] << " " << P[1] << endl;
    return 0;
}
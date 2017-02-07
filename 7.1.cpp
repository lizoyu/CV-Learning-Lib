/*************************************************************************
	> File Name: 7.1.cpp - EM for mixtures of Gaussians
	> Author: Chungyuen Li
	> Mail: lizoyu1@gmail.com
	> Created Time: 2017年02月05日 星期日 11时19分10秒
 ************************************************************************/

#include<iostream>
#include<Eigen/Dense>
using namespace std;
using namespace Eigen;

void EMforMoG( MatrixXd &trainData, int num_clusters )
{
    // lambda: (K,1); mean: (trainData.rows(),K); var:(rows*K,K);
    // initiate
    VectorXd lambda(K);
    lambda += 1/K;
    MatrixXd mean(trainData.rows(),K);
    for( int k = 0; k < K; ++k )
    {
        for( int i = 0; i < trainData.rows(); ++i )
            mean(i,k) = trainData.row(i).head(k+1) / (k+1);
    }
    MatrixXd var(trainData.rows()*K,trainData.rows());
    VectorXd mean_all(trainData.rows());
    for( int i = 0; i < mean_all.size(); ++i )
        mean_all(i) = trainData.row(i).mean();
    for( int i = 0; i < trainData.cols(); ++i )
    {
        for( int j = 0; j < trainData.rows(); ++j )
            var(j,j) += pow( trainData(j,i) - mean_all(j), 2 );
    }
    
    // repeat until L dont change
    while( L - L_old != 0 )
    {
        // E-step
        MatrixXd l(K,trainData.cols());
        MatrixXd r(K,trainData.cols());
        for( int i = 0; i < trainData.cols(); ++i )
        {
            // likelihood
            for( int k = 0; k < K; ++k )
            {
                VectorXd deviation = trainData.cols(i) - mean(k);
                MatrixXd var_k = var.block( k*trainData.rows() , 0,
                                        trainData.rows(), trainData.rows() );
                l(k,i) = exp( -0.5*deviation.transpose()*var_k.inverse()
                             *deviation) / ( pow( 2*M_PI, 
                round(trainData.rows()/2) )*sqrt( abs(var_k.determinant() ) )); 
            }
            // posterior
            r.col(i) = l.col(i) / l.col(i).sum();
        }

        // M-step
        
    }   

}

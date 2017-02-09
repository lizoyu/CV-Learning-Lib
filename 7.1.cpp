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
        for( int k = 0; k < K; ++k )
        {
            lambda(k) = r.row(k).sum() / r.sum();
            mean.col(k).empty();
            for( int i = 0; i < r.cols(); ++i )
                mean += r(k,i) * trainData.col(i);
            mean.col(k) = mean.col(k) / r.row(k).sum();
            for( int i = 0; i < r.cols(); ++i )
            {
                VectorXd deviation = trainData.col(i) - mean.col(k);
                var.block(k*mean.rows(),0,mean.rows(),mean.rows()).empty(); 
                var.block(k*mean.rows(),0,mean.rows(),mean.rows()) += 
                    r(k,i) * deviation * deviation.transpose(); 
            }
            var.block(k*mean.rows(),0,mean.rows(),mean.rows()) =  
                var.block(k*mean.rows(),0,mean.rows(),mean.rows()).diagonal()
                .asDiagonal() / r.row(k).sum();
        }
        // Log likelihood and EM bound
        double L, B;
        for( int i = 0; i < trainData.cols(); ++i )
        {
            double temp_L = 0;
            for( int k = 0; k < K; ++k )
            {
                VectorXd deviation = trainData.cols(i) - mean(k);
                MatrixXd var_k = var.block( k*trainData.rows() , 0,
                                        trainData.rows(), trainData.rows() );
                temp_L += lambda(k) * exp( -0.5*deviation.transpose()*
                    var_k.inverse() *deviation) / ( pow( 2*M_PI,round(
                    trainData.rows()/2) )*sqrt( abs( var_k.determinant() ) ) );
                B += r(k,i)*log10((lambda(k)*exp(-0.5*deviation.transpose()*
                    var_k.inverse()*deviation) / (pow(2*M_PI,round(
                    trainData.rows()/2))*sqrt(abs(var_k.determinantmZ))))/r(k,i));
            }
            L += log10( temp );
        }
    }   
}

int main()
{
    
    return 0;
}

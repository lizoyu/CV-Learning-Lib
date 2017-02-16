/*************************************************************************
	> File Name: 7.1.cpp - EM for mixtures of Gaussians
	> Author: Chungyuen Li
	> Mail: lizoyu1@gmail.com
	> Created Time: 2017年02月05日 星期日 11时19分10秒
 ************************************************************************/

#include<iostream>
#include<vector>
#include<Eigen/Dense>
#include<Eigen/QR>
#include<Eigen/Eigenvalues>
using namespace std;
using namespace Eigen;

void EMforMoG( MatrixXd &trainData, VectorXd &lambda, MatrixXd &mean,
              MatrixXd &var, int num_clusters )
{
    // lambda: (K,1); mean: (trainData.rows(),K); var:(rows*K,rows);
    // initiate
    int K = num_clusters;
    lambda += VectorXd::Ones(K) / K;
    for( int k = 0; k < K; ++k )
    {
        for( int i = 0; i < trainData.rows(); ++i )
            mean(i,k) = trainData.row(i).head(k+1).sum() / (k+1);
    }
    VectorXd mean_all(trainData.rows());
    for( int i = 0; i < mean_all.size(); ++i )
        mean_all(i) = trainData.row(i).mean();
    for( int i = 0; i < trainData.cols(); ++i )
    {
        for( int j = 0; j < trainData.rows(); ++j )
            var(j,j) += pow( trainData(j,i) - mean_all(j), 2 );
    }
    for( int i = 1; i < K; ++i )
        var.block( i*trainData.rows(), 0, trainData.rows(), trainData.rows() ) = 
            var.block( 0, 0, trainData.rows(), trainData.rows() );
    // repeat until L dont change
    double L = 0, L_old = 1;
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
                VectorXd deviation = trainData.col(i) - mean.col(k);
                MatrixXd var_k = var.block( k*trainData.rows() , 0,
                                        trainData.rows(), trainData.rows() );
                CompleteOrthogonalDecomposition<MatrixXd> cod( var_k );
                SelfAdjointEigenSolver<MatrixXd> es( var_k );
                l(k,i) = lambda(k)*exp(-0.5*deviation.transpose()*
                    cod.pseudoInverse()*deviation) / (pow(2*M_PI, 
                round(trainData.rows()/2))*sqrt(abs(es.eigenvalues().sum()))); 
            }
            // posterior
            r.col(i) = l.col(i) / l.col(i).sum();
        }
        cout << "posterior:" << endl << r << endl;
        // M-step
        for( int k = 0; k < K; ++k )
        {
            lambda(k) = r.row(k).sum() / r.sum();
            mean.col(k) = VectorXd::Zero( mean.rows() );
            for( int i = 0; i < r.cols(); ++i )
                mean.col(k) += r(k,i) * trainData.col(i);
            mean.col(k) = mean.col(k) / r.row(k).sum();
            for( int i = 0; i < r.cols(); ++i )
            {
                VectorXd deviation = trainData.col(i) - mean.col(k);
                var.block(k*mean.rows(),0,mean.rows(),mean.rows()) = 
                    MatrixXd::Zero( mean.rows(), mean.rows() ); 
                var.block(k*mean.rows(),0,mean.rows(),mean.rows()) += 
                    r(k,i) * deviation * deviation.transpose(); 
            }
            VectorXd vec_d = var.block(
                k*mean.rows(),0,mean.rows(),mean.rows()).diagonal();
            MatrixXd var_d = vec_d.asDiagonal();
            var.block(k*mean.rows(),0,mean.rows(),mean.rows()) = 
                var_d / r.row(k).sum();
        }
        cout << "lambda:" << endl << lambda << endl;
        cout << "mean:" << endl << mean << endl;
        cout << "var:" << endl << var << endl;

        // Log likelihood and EM bound
        double B;
        L_old = L;
        for( int i = 0; i < trainData.cols(); ++i )
        {
            double temp_L = 0;
            for( int k = 0; k < K; ++k )
            {
                VectorXd deviation = trainData.col(i) - mean.col(k);
                MatrixXd var_k = var.block( k*trainData.rows() , 0,
                                        trainData.rows(), trainData.rows() );
                CompleteOrthogonalDecomposition<MatrixXd> cod( var_k );
                SelfAdjointEigenSolver<MatrixXd> es( var_k );
                temp_L += lambda(k) * exp( -0.5*deviation.transpose()*
                    cod.pseudoInverse() *deviation) / ( pow( 2*M_PI,round(
                    trainData.rows()/2) )*sqrt( abs( es.eigenvalues().sum() ) ));
                B += r(k,i)*log10((lambda(k)*exp(-0.5*deviation.transpose()*
                    cod.pseudoInverse()*deviation) / (pow(2*M_PI,round(
                    trainData.rows()/2))*sqrt(abs(es.eigenvalues().sum()))))/
                    r(k,i));
            }
            L += log10( temp_L );
        }
        cout << "L: " << L << endl;
    }   
}

double inferMoG( VectorXd &lambda, MatrixXd &mean, MatrixXd &var, VectorXd point,
	       double prior )
{
    int K = lambda.size();
    // compute likelihood for new point
    double l;
    for( int k = 0; k < K; ++k )
    {
        VectorXd deviation = point - mean.col(k);
        MatrixXd var_k = var.block( k*mean.rows() , 0,
                                mean.rows(), mean.rows() );
        l += lambda(k) * exp( -0.5*deviation.transpose()*var_k.inverse()
                     *deviation) / ( pow( 2*M_PI, 
        round(mean.rows()/2) )*sqrt( abs(var_k.determinant() ) )); 
    }
    l *= prior;

    return l;
}

int main()
{
    // two class, 1 - (10, 20, 30); 2 - (70, 80, 90);
    int num_class = 2;
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
	point << 68, 78, 88;

    // train two MoG model for each class
    int K = 3;
    VectorXd lambda_1(K), lambda_2(K);
    MatrixXd mean_1(trainData_1.rows(),K), mean_2(trainData_2.rows(),K);
    MatrixXd var_1(K*mean_1.rows(),mean_1.rows()), 
        var_2(K*mean_2.rows(),mean_2.rows());
    double prior_1 = 0.5, prior_2 = 0.5;
    EMforMoG( trainData_1, lambda_1, mean_1, var_1, K );
    EMforMoG( trainData_2, lambda_2, mean_2, var_2, K );

    // inference
    double l_1 = inferMoG( lambda_1, mean_1, var_1, point, prior_1 );
    double l_2 = inferMoG( lambda_2, mean_2, var_2, point, prior_2 );
    vector<double> P(2);
    P[0] = l_1 / (l_1 + l_2);
    P[1] = l_2 / (l_1 + l_2);

    cout << "Probility: " << P[0] << " " << P[1] << endl;
    return 0;
}
